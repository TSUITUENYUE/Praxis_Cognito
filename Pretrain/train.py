import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import os, math
import genesis as gs
from rsl_rl.modules import EmpiricalNormalization

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
        self.normalizer_path = config.normalizer_path
        self.batch_size  = config.batch_size
        self.device      = config.device
        self.num_epochs  = config.num_epochs
        self.grad_clip_value = 1.0
        self.obs_dim = rl_config.obs.num_obs

        # parse filename layout (unchanged)
        filename = os.path.basename(self.load_path)
        parts = filename.rstrip('.h5').strip().split()
        num_envs = int(parts[0]); self.episodes = int(parts[1])
        self.max_episode_seconds = int(parts[2]); self.frame_rate = int(parts[3])
        self.max_episode_len = self.max_episode_seconds * self.frame_rate

        # model & geometry
        self.vae = torch.compile(model.to(self.device))
        self.agent = self.vae.agent
        self.fk_model = self.agent.fk_model.to(self.device)

        gs.init(logging_level="warning")
        self.env = Go2Env(
            num_envs=self.batch_size,
            env_cfg=rl_config.env,
            obs_cfg=rl_config.obs,
            reward_cfg=rl_config.reward,
            command_cfg=rl_config.command,
            show_viewer=False,
            agent=self.agent,
        )

        self.end_effector_indices = self.agent.end_effector
        self.edge_index = build_edge_index(self.fk_model, self.end_effector_indices, self.device)

        # dataset
        self.dataset = TrajectoryDataset(
            processed_path=config.processed_path,
            source_path=self.load_path,
            agent=self.agent
        )

        if rl_config.train.empirical_normalization:
            state = torch.load(self.normalizer_path, map_location="cpu")
            #print(state['obs_norm'])
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.obs_normalizer.load_state_dict(state["obs_norm"])
            self.obs_normalizer.to(self.device)
            self.obs_normalizer.eval()
            for p in self.obs_normalizer.parameters():
                p.requires_grad_(False)
            '''
            self.critic_obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.critic_obs_normalizer.load_state_dict(state["critic_obs_norm"])
            self.critic_obs_normalizer.to(self.device)
            self.critic_obs_normalizer.eval()
            for p in self.critic_obs_normalizer.parameters():
                p.requires_grad_(False)
            '''
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)
        # push dataset normalization into VAE + decoder buffers
        pm = self.dataset.pos_mean.to(self.device); ps = self.dataset.pos_std.to(self.device)
        with torch.no_grad():
            self.vae.pos_mean.copy_(pm)
            self.vae.pos_std.copy_(ps)

        # optimizer & schedulers
        base_lr = config.optimizer.lr
        self.optimizer = optim.AdamW(self.vae.parameters(), lr=base_lr, weight_decay=1e-5, fused=True)
        num_total_steps = self.num_epochs * max(1, (self.episodes // self.batch_size))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_total_steps)
        self.scaler = GradScaler()

        # weights
        self.strategy = config.beta_anneal.strategy
        self.warm_up  = config.beta_anneal.warm_up
        self.max_beta = config.beta_anneal.max_beta

        self.lambda_kinematic = config.kino.lambda_kinematic
        self.lambda_dynamic   = config.kino.lambda_dynamic

        self.lambda_sim = config.sim.lambda_sim
        self.w_q  = config.sim.w_q
        self.w_dq = config.sim.w_dq
        self.w_obs= config.sim.w_obs

    # ---------- helpers ----------
    @staticmethod
    def masked_mse(pred, target, mask):
        # pred/target: [B,T,D], mask: [B,T,1]
        per_t = ((pred - target) ** 2).mean(dim=-1)  # [B,T]
        denom = mask.sum().clamp_min(1.0)
        return (per_t * mask.squeeze(-1)).sum() / denom

    def _norm_seq(self, x):
        """x: [B,T,obs_dim] -> normalized with EmpiricalNormalization (no stat updates)."""
        B, T, D = x.shape
        x_flat = x.reshape(B*T, D)
        x_n = self.obs_normalizer(x_flat)  # module in eval mode => uses stored stats
        return x_n.view(B, T, D)


    # ---------- training ----------
    def train(self):
        torch.set_float32_matmul_precision('high')
        self.vae.train()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=True
        )

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
                obs     = obs.to(self.device)              # [B,T,obs_dim] (UN-normalized from dataset)
                act     = act.to(self.device)
                q       = q.to(self.device)
                dq      = dq.to(self.device)
                mask    = mask.to(self.device)[:, :, None] # [B,T,1]

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    beta = get_beta(global_step, total_steps, strategy=self.strategy,
                                    warmup_epochs=self.warm_up, max_beta=self.max_beta)

                    # teacher forcing ratio: decay to 0 by midway
                    tf_ratio = max(0.0, 1.0 - global_step / (0.5 * total_steps))

                    # --- forward (policy decoder returns actions) ---
                    recon_mu, joint_cmd, actions_seq, log_sigma, *aux = self.vae(
                        graph_x, self.edge_index, mask, self.obs_normalizer,
                        obs_seq=obs,
                        q=q,
                        dq=dq,
                        tf_ratio=tf_ratio,
                    )
                    # --------------- Genesis rollout (teacher for surrogate) ---------------
                    with torch.no_grad():
                        reset_out = self.env.reset()
                        obs0_raw = reset_out[0] if isinstance(reset_out, tuple) else reset_out  # [B,obs_dim]
                        q0  = self.env.robot.get_dofs_position(self.env.motors_dof_idx)  # [B,d]
                        dq0 = self.env.robot.get_dofs_velocity(self.env.motors_dof_idx)  # [B,d]

                        obs0_n = self.obs_normalizer(obs0_raw.to(self.device))

                        obs_sim_n_seq, q_sim_seq, dq_sim_seq = [], [], []
                        T = actions_seq.shape[1]

                        for t in range(T):
                            a_t = actions_seq[:, t, :].detach()  # do not backprop into policy via sim
                            obs_t_raw, rew_t, done_t, info_t = self.env.step(a_t)
                            q_t  = self.env.robot.get_dofs_position(self.env.motors_dof_idx)
                            dq_t = self.env.robot.get_dofs_velocity(self.env.motors_dof_idx)

                            obs_t_n = self.obs_normalizer(obs_t_raw.to(self.device))

                            obs_sim_n_seq.append(obs_t_n)
                            q_sim_seq.append(q_t.to(self.device))
                            dq_sim_seq.append(dq_t.to(self.device))

                        obs_sim_n_seq = torch.stack(obs_sim_n_seq, dim=1)  # [B,T,obs_dim] (normalized)
                        q_sim_seq     = torch.stack(q_sim_seq, dim=1)      # [B,T,d]
                        dq_sim_seq    = torch.stack(dq_sim_seq, dim=1)     # [B,T,d]

                        q0  = q0.to(self.device)
                        dq0 = dq0.to(self.device)

                    # --------------- Surrogate rollout (inside the VAE) ---------------
                    sur_obs_seq, sur_q_seq, sur_dq_seq = self.vae.predict_dynamics(
                        actions_seq.detach(),    # [B,T,d] — only surrogate learns from sim loss
                        obs0=obs0_n.detach(),
                        q0=q0.detach(),
                        dq0=dq0.detach(),
                        mask=mask,
                    )

                    # --- VAE loss (pose + action + KL) ---
                    loss, kinematic_loss, dynamic_loss, kl_loss = self.vae.loss(
                        recon_mu, log_sigma, graph_x, act, actions_seq, mask, *aux,
                        beta=beta,
                        lambda_kinematic=self.lambda_kinematic,
                        lambda_dynamic=self.lambda_dynamic
                    )

                    # --- Sim loss (only trains surrogate via surrogate path) ---
                    q_loss  = self.masked_mse(sur_q_seq,  q_sim_seq.detach(),  mask)
                    dq_loss = self.masked_mse(sur_dq_seq, dq_sim_seq.detach(), mask)
                    obs_loss = self.masked_mse(sur_obs_seq, obs_sim_n_seq.detach(), mask)
                    sim_loss = self.w_q * q_loss + self.w_dq * dq_loss + self.w_obs * obs_loss

                    total_loss = loss + self.lambda_sim * sim_loss

                    # --- diag: unnormalized recon MSE in world units ---
                    unnormalized_recon_mu = _denorm_positions(recon_mu, self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_graph_x  = _denorm_positions(graph_x.reshape(recon_mu.shape), self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_loss = self.masked_mse(unnormalized_recon_mu, unnormalized_graph_x, mask)

                # Backprop THROUGH total_loss so surrogate learns from sim_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                print(
                    f"Batch {i + 1}/{len(dataloader)}, "
                    f"Total:{total_loss.item():.4f} | VAE:{loss.item():.4f} "
                    f"| Kin:{kinematic_loss.item():.4f} | Unnorm_Kin:{unnormalized_loss.item():.4f} "
                    f"| Dyn:{dynamic_loss.item():.4f} | KL:{kl_loss.item():.4f} "
                    f"| Sim:{sim_loss.item():.4f} (q={q_loss.item():.4f}, dq={dq_loss.item():.4f}, obs={obs_loss.item():.4f}) "
                    f"| Beta:{beta:.3f} | TF:{tf_ratio:.2f}"
                )

            os.makedirs(self.save_path, exist_ok=True)
            save_path = os.path.join(self.save_path, f"vae_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(self.vae.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")
