import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import os, math
import genesis as gs
from .go2_env import Go2Env
from .dataset import TrajectoryDataset
from .utils import build_edge_index

def get_beta(epoch, total_epochs, strategy='cyclical', num_cycles=4, max_beta=1.0, warmup_epochs=20):
    if strategy == 'warmup':
        return 0.0 if epoch < warmup_epochs else max_beta * (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    elif strategy == 'cyclical':
        cycle_length = max(1, total_epochs // num_cycles)
        cycle_progress = (epoch % cycle_length) / cycle_length
        return max_beta * (1 / (1 + math.exp(-10 * (cycle_progress - 0.5))))
    else:
        return max_beta

class Trainer:
    def __init__(self, model, rl_config, config):
        self.load_path   = config.load_path
        self.save_path   = config.save_path
        self.batch_size  = config.batch_size
        self.device      = config.device
        self.num_epochs  = config.num_epochs
        self.grad_clip_value = 1.0
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = (
            rl_config.env_cfg, rl_config.obs_cfg, rl_config.reward_cfg, rl_config.command_cfg, rl_config.train_cfg
        )

        # parse filename layout if you rely on it
        filename = os.path.basename(self.load_path)
        parts = filename.rstrip('.h5').strip().split()
        num_envs = int(parts[0]); self.episodes = int(parts[1])
        self.max_episode_seconds = int(parts[2]); self.frame_rate = int(parts[3])
        self.max_episode_len = self.max_episode_seconds * self.frame_rate

        # model & geometry
        self.vae = model.to(self.device)
        self.vae_prior = self.vae.prior
        self.vae = torch.compile(self.vae).to(self.device)
        self.agent = self.vae.agent
        self.fk_model = self.agent.fk_model.to(self.device)

        gs.init()
        self.env = Go2Env(
            num_envs=self.batch_size,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            agent=self.agent,
        )

        self.end_effector_indices = self.agent.end_effector
        self.edge_index = build_edge_index(self.fk_model, self.end_effector_indices, self.device)

        # dataset
        self.dataset = TrajectoryDataset(processed_path=config.processed_path, source_path=self.load_path, agent=self.agent)

        # push dataset normalization into VAE + decoder buffers
        pm = self.dataset.pos_mean.to(self.device); ps = self.dataset.pos_std.to(self.device)
        with torch.no_grad():
            self.vae.pos_mean.copy_(pm); self.vae.pos_std.copy_(ps)

        # optimizer (give policy head a bit more LR)
        base_lr = config.optimizer.lr

        self.optimizer = optim.AdamW(params=self.vae.parameters(), lr=base_lr ,weight_decay=1e-5,fused=True,)

        num_total_steps = self.num_epochs * max(1, (self.episodes // self.batch_size))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_total_steps)
        self.scaler = GradScaler()

        # beta schedule cfg
        self.strategy = config.beta_anneal.strategy
        self.warm_up  = config.beta_anneal.warm_up
        self.max_beta = config.beta_anneal.max_beta

        self.lambda_kinematic = config.kino.lambda_kinematic
        self.lambda_dynamic = config.kino.lambda_dynamic

        self.lambda_sim = config.sim.lambda_sim  # total weight on sim loss
        self.w_q = config.sim.w_q
        self.w_dq = config.sim.w_dq
        self.w_obs = config.sim.w_obs

    # Add to Trainer class
    @staticmethod
    def masked_mse(pred, target, mask):
        """
        pred, target: [B, T, D]
        mask: [B, T, 1] float {0,1}
        """
        diff2 = (pred - target) ** 2
        per_t = diff2.mean(dim=-1)  # [B, T]
        valid = mask.sum().clamp_min(1.0)  # scalar
        return (per_t * mask.squeeze(-1)).sum() / valid

    def train(self):
        torch.set_float32_matmul_precision('high')
        self.vae.train()

        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=True,
                                drop_last=True)

        total_steps = self.num_epochs * len(dataloader)

        def _denorm_positions(traj_norm, pos_mean, pos_std):
            B, T, D = traj_norm.shape
            num_nodes = D // 3
            x = traj_norm.reshape(B, T, num_nodes, 3)
            x = x * pos_std[None, None, None, :].to(self.device) + pos_mean[None, None, None, :].to(self.device)
            return x.reshape(B, T, D)

        for epoch in range(self.num_epochs):
            for i, (graph_x, obs, act, q, dq, mask) in enumerate(dataloader):
                global_step = epoch * len(dataloader) + i
                graph_x = graph_x.to(self.device)
                obs = obs.to(self.device)
                act = act.to(self.device)
                q = q.to(self.device)
                dq = dq.to(self.device)
                mask = mask.to(self.device)[:,:,None]
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    beta = get_beta(global_step, total_steps, strategy=self.strategy,
                                    warmup_epochs=self.warm_up, max_beta=self.max_beta)

                    # teacher forcing ratio: decay to 0 by midway
                    tf_ratio = max(0.0, 1.0 - global_step / (0.5 * total_steps))

                    # --- forward (policy decoder returns actions) ---
                    recon_mu, joint_cmd, actions_seq, log_sigma, *aux = self.vae(
                        graph_x, self.edge_index, mask,
                        obs_seq=obs,
                        q=q,
                        dq=dq,
                        tf_ratio=tf_ratio,
                    )
                    # --------------- Genesis rollout (teacher for surrogate) ---------------
                    with torch.no_grad():
                        # reset env to a known start per batch; capture initial state
                        reset_out = self.env.reset()
                        if isinstance(reset_out, tuple):
                            obs0 = reset_out[0]
                        else:
                            obs0 = reset_out
                        # get initial q0,dq0 from sim
                        q0 = self.env.robot.get_dofs_position(self.env.motors_dof_idx)  # [B, d]
                        dq0 = self.env.robot.get_dofs_velocity(self.env.motors_dof_idx)  # [B, d]

                        obs_sim_seq, q_sim_seq, dq_sim_seq = [], [], []
                        T = actions_seq.shape[1]
                        for t in range(T):
                            # use DETACHED policy actions so gradients don't flow into policy via sim loss
                            a_t = actions_seq[:, t, :].detach()
                            obs_t, rew_t, done_t, info_t = self.env.step(a_t)

                            q_t = self.env.robot.get_dofs_position(self.env.motors_dof_idx)
                            dq_t = self.env.robot.get_dofs_velocity(self.env.motors_dof_idx)

                            obs_sim_seq.append(obs_t.to(self.device))
                            q_sim_seq.append(q_t.to(self.device))
                            dq_sim_seq.append(dq_t.to(self.device))

                        obs_sim_seq = torch.stack(obs_sim_seq, dim=1)  # [B, T, obs_dim]
                        q_sim_seq = torch.stack(q_sim_seq, dim=1)  # [B, T, d]
                        dq_sim_seq = torch.stack(dq_sim_seq, dim=1)  # [B, T, d]
                        # also keep the initial state for the surrogate
                        q0 = q0.to(self.device);
                        dq0 = dq0.to(self.device);
                        obs0 = obs0.to(self.device)

                    # --------------- Surrogate rollout (inside the VAE) ---------------
                    sur_obs_seq = sur_q_seq = sur_dq_seq = None
                    # IMPORTANT: feed DETACHED actions so sim loss only updates surrogate params
                    sur_obs_seq, sur_q_seq, sur_dq_seq = self.vae.predict_dynamics(
                        actions_seq.detach(),  # [B,T,d]
                        obs0=obs0.detach(),
                        q0=q0.detach(),
                        dq0=dq0.detach(),
                        mask=mask,  # [B,T,1]
                    )


                    # --- main loss: heteroscedastic NLL on normalized graph positions ---
                    loss, kinematic_loss, dynamic_loss, kl_loss = self.vae.loss(
                        recon_mu, log_sigma, graph_x, act, actions_seq, mask, *aux,
                        beta=beta,
                        lambda_kinematic=self.lambda_kinematic,
                        lambda_dynamic=self.lambda_dynamic
                    )
                    # --- Sim loss (only trains surrogate) ---
                    sim_loss = torch.tensor(0.0, device=self.device)
                    if (sur_obs_seq is not None) and (sur_q_seq is not None) and (sur_dq_seq is not None):
                        # Mask-aware MSE vs Genesis rollouts
                        q_loss = self.masked_mse(sur_q_seq, q_sim_seq, mask)
                        dq_loss = self.masked_mse(sur_dq_seq, dq_sim_seq, mask)
                        # If obs dims don’t match exactly you can drop or subselect here
                        obs_loss = self.masked_mse(sur_obs_seq, obs_sim_seq, mask) if sur_obs_seq.shape[-1] == \
                                                                                      obs_sim_seq.shape[-1] else 0.0
                        sim_loss = self.w_q * q_loss + self.w_dq * dq_loss + self.w_obs * obs_loss

                    total_loss = loss + self.lambda_sim * sim_loss
                    # --- diag: unnormalized recon MSE in world units ---
                    unnormalized_recon_mu = _denorm_positions(recon_mu, self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_graph_x  = _denorm_positions(graph_x.reshape(recon_mu.shape), self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_loss = self.masked_mse(unnormalized_recon_mu, unnormalized_graph_x, mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                print(
                    f"Batch {i + 1}/{len(dataloader)}, "
                    f"Total:{total_loss.item():.4f} | VAE:{loss.item():.4f} "
                    f"| Kin:{kinematic_loss.item():.4f} | Unnorm_Kin:{unnormalized_loss.item():.4f} | Dyn:{dynamic_loss.item():.4f} | KL:{kl_loss.item():.4f} "
                    f"| Sim:{sim_loss.item():.4f} (q={q_loss:.4f}, dq={dq_loss:.4f}, obs={obs_loss:.4f}) "
                    f"| Beta:{beta:.3f} | TF:{tf_ratio:.2f}"
                )

            # (optional) save per epoch
            os.makedirs(self.save_path, exist_ok=True)
            save_path = os.path.join(self.save_path, f"vae_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(self.vae.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")
