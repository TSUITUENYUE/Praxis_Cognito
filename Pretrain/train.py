import torch
import torch.nn as nn
import torch.optim as optim

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

def masked_quat_geodesic(pred_w, tgt_w, mask, eps=1e-8):
    # pred_w, tgt_w: [B,T,4] (wxyz, normalized)
    # mask: [B,T,1] or [B,T]
    # geodesic distance: 2*acos(|<q1,q2>|); we can use (1 - dot^2) as a smooth proxy
    pred = pred_w / (pred_w.norm(dim=-1, keepdim=True) + eps)
    tgt  = tgt_w  / (tgt_w.norm(dim=-1, keepdim=True) + eps)
    dot  = (pred * tgt).sum(dim=-1).abs().clamp(max=1.0)       # [B,T]
    # proxy loss (smooth, scale ~ angle^2 near 0)
    loss = 1.0 - dot**2
    if mask.dim() == 3: mask = mask.squeeze(-1)
    # average over unmasked
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


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
        #self.vae = torch.compile(model.to(self.device),mode="reduce-overhead")
        self.vae = model.to(self.device)
        self.vae.encoder = torch.compile(self.vae.encoder,mode="reduce-overhead")
        #self.vae.decoder = torch.compile(self.vae.decoder,mode="reduce-overhead")
        self.vae.surrogate = torch.compile(self.vae.surrogate,mode="reduce-overhead")
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

            self.critic_obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.critic_obs_normalizer.load_state_dict(state["critic_obs_norm"])
            self.critic_obs_normalizer.to(self.device)
            self.critic_obs_normalizer.eval()
            for p in self.critic_obs_normalizer.parameters():
                p.requires_grad_(False)

        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            for p in self.obs_normalizer.parameters():
                p.requires_grad_(False)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)
            for p in self.critic_obs_normalizer.parameters():
                p.requires_grad_(False)
        # push dataset normalization into VAE + decoder buffers
        pm = self.dataset.pos_mean.to(self.device); ps = self.dataset.pos_std.to(self.device)
        with torch.no_grad():
            self.vae.pos_mean.copy_(pm)
            self.vae.pos_std.copy_(ps)
        # in __init__ (after you computed pm/ps and copied into buffers)
        self.pos_mean_d = self.dataset.pos_mean.to(self.device)
        self.pos_std_d = self.dataset.pos_std.to(self.device)

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
        self.lambda_dynamic   = config.kino.lambda_dynamic / self.agent.n_dofs

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
        self.vae.train()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True
        )

        total_steps = self.num_epochs * len(dataloader)

        def _denorm_positions(traj_norm, pos_mean_d, pos_std_d):
            # traj_norm: [B,T,3*num_nodes]; pos_mean_d/pos_std_d already on device
            B, T, D = traj_norm.shape
            num_nodes = D // 3
            x = traj_norm.reshape(B, T, num_nodes, 3)
            x = x * pos_std_d[None, None, None, :] + pos_mean_d[None, None, None, :]
            return x.reshape(B, T, D)

        for epoch in range(self.num_epochs):
            for i, (x, q, dq, p, dp, dw, obs, act, mask) in enumerate(dataloader):
                global_step = epoch * len(dataloader) + i

                x      = x.to(self.device)

                q_gt   = q.to(self.device)
                dq_gt  = dq.to(self.device)

                p_gt   = p.to(self.device)
                dp_gt  = dp.to(self.device)
                dw_gt   = dw.to(self.device)

                obs_gt = obs.to(self.device)              # [B,T,obs_dim] (UN-normalized from dataset)
                act_gt = act.to(self.device)

                mask   = mask.to(self.device)[:, :, None] # [B,T,1]

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    beta = get_beta(global_step, total_steps, strategy=self.strategy,
                                    warmup_epochs=self.warm_up, max_beta=self.max_beta)

                    # teacher forcing ratio: decay to 0 by midway

                    tf_ratio = max(0.0, 1.0 - global_step / (0.5 * total_steps))

                    # --- forward (policy decoder returns actions) ---
                    traj_pred, state_pred, act_pred, obs_pred, *aux = self.vae(
                         x, self.edge_index, mask,           # encoder_input
                         self.obs_normalizer,                # normalizer for obs
                         obs_seq=obs_gt,                     # for decoder teacher forcing
                         q=q_gt, dq=dq_gt,
                         p=p_gt, dp=dp_gt, dw=dw_gt,
                         tf_ratio=tf_ratio,
                    )

                    x_recon = traj_pred['graph']

                    q_seq = state_pred['q']
                    dq_seq = state_pred['dq']

                    p_seq = state_pred['p']
                    dp_seq = state_pred['dp']
                    w_seq = state_pred['w']
                    dw_seq = state_pred['dw']

                    act_seq = act_pred['act']
                    mu_seq = act_pred['mu']
                    log_std_seq = act_pred['log_std']
                    log_sigma = act_pred['log_sigma']

                    obs_seq = obs_pred['obs']
                    u_seq = obs_seq[:,:,-6:-3]
                    du_seq = obs_seq[:,:,-3:]

                    z = aux[0]
                    # --------------- Genesis rollout (teacher for surrogate) ---------------
                    with torch.inference_mode():
                        was_training = self.vae.decoder.training
                        self.vae.decoder.eval()
                        try:
                            reset_out = self.env.reset()
                            obs0_raw = reset_out[0] if isinstance(reset_out, tuple) else reset_out  # [B,obs_dim]
                            obs_t_n = self.obs_normalizer(obs0_raw.to(self.device, non_blocking=True))

                            B = act_seq.shape[0]
                            T = act_seq.shape[1]
                            # Pre-allocate on device with correct dtypes
                            q_sim_seq  = torch.empty(size=q_seq.shape, device=self.device, dtype=obs_t_n.dtype)
                            dq_sim_seq = torch.empty(size=q_seq.shape, device=self.device, dtype=obs_t_n.dtype)

                            p_sim_seq  = torch.empty(size=p_seq.shape, device=self.device, dtype=obs_t_n.dtype)
                            dp_sim_seq = torch.empty(size=dp_seq.shape, device=self.device, dtype=obs_t_n.dtype)

                            w_sim_seq  = torch.empty(size=w_seq.shape,device=self.device, dtype=obs_t_n.dtype)
                            dw_sim_seq = torch.empty(size=dw_seq.shape, device=self.device, dtype=obs_t_n.dtype)

                            obs_sim_seq = torch.empty(size=obs_seq.shape, device=self.device, dtype=obs_t_n.dtype)

                            for t in range(T):
                                mask_t = mask[:, t, :]
                                a_t, _, _ = self.vae.decoder(z, obs_t_n, mask_t)
                                obs_t_raw, rew_t, done_t, info_t = self.env.step(a_t)

                                q_t = self.env.dof_pos
                                dq_t = self.env.dof_vel
                                p_t = self.env.base_pos
                                dp_t = self.env.base_lin_vel
                                w_t = self.env.base_quat
                                dw_t = self.env.base_ang_vel

                                obs_t_n = self.obs_normalizer(obs_t_raw.to(self.device, non_blocking=True))

                                q_sim_seq[:,t].copy_(q_t)
                                dq_sim_seq[:,t].copy_(dq_t)

                                p_sim_seq[:,t].copy_(p_t)
                                dp_sim_seq[:,t].copy_(dp_t)
                                w_sim_seq[:,t].copy_(w_t)
                                dw_sim_seq[:, t].copy_(dw_t)

                                obs_sim_seq[:, t].copy_(obs_t_n)


                            u_sim_seq = obs_sim_seq[:, :, -6:-3]
                            du_sim_seq = obs_sim_seq[:, :, -3:]
                        finally:
                            self.vae.decoder.train(was_training)
                    # --- VAE loss (pose + action + KL) ---
                    loss, kinematic_loss, dynamic_loss, kl_loss = self.vae.loss(
                        x_recon, log_sigma, x,
                        act_gt, act_seq,
                        mu_seq, log_std_seq,
                        mask, *aux,
                        beta=beta,
                        lambda_kinematic=self.lambda_kinematic,
                        lambda_dynamic=self.lambda_dynamic
                    )

                    # state-space losses (detach sim targets)
                    q_loss = self.masked_mse(q_seq, q_sim_seq.detach(), mask)
                    dq_loss = self.masked_mse(dq_seq, dq_sim_seq.detach(), mask)
                    p_loss = self.masked_mse(p_seq, p_sim_seq.detach(), mask)
                    dp_loss = self.masked_mse(dp_seq, dp_sim_seq.detach(), mask)
                    dw_loss = self.masked_mse(dw_seq, dw_sim_seq.detach(), mask)
                    w_loss = masked_quat_geodesic(w_seq, w_sim_seq.detach(), mask)

                    u_loss = self.masked_mse(u_seq, u_sim_seq.detach(), mask)
                    du_loss = self.masked_mse(du_seq, du_sim_seq.detach(), mask)

                    state_loss = (
                            self.w_q * q_loss +
                            self.w_dq * dq_loss +
                            self.w_p * p_loss +
                            self.w_dp * dp_loss +
                            self.w_w * w_loss +
                            self.w_dw * dw_loss +
                            self.w_u * u_loss +
                            self.w_du * du_loss
                    )

                    # optional: obs regularizer (use your exact normalized obs on both sides)
                    # obs_pred_n is from surrogate’s predict_dynamics; obs_sim_n_seq from Genesis → normalized
                    obs_reg = self.masked_mse(obs_seq, obs_sim_seq.detach(), mask)
                    sim_loss = state_loss + self.w_obs * obs_reg  # choose small self.w_obs, e.g., 0.1 * (avg state weight)

                    total_loss = loss + self.lambda_sim * sim_loss

                    # --- diag: unnormalized recon MSE in world units ---
                    unnormalized_recon_mu = _denorm_positions(x_recon, self.pos_mean_d, self.pos_std_d)
                    unnormalized_graph_x  = _denorm_positions(x.reshape(x_recon.shape), self.pos_mean_d, self.pos_std_d)
                    unnormalized_loss = self.masked_mse(unnormalized_recon_mu, unnormalized_graph_x, mask)

                # Backprop THROUGH total_loss so surrogate learns from sim_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                log_every = 1

                if (global_step % log_every == 0) or (i == len(dataloader) - 1):
                    print(
                        f"Batch {i + 1}/{len(dataloader)}, "
                        f"Total:{total_loss.item():.4f} | VAE:{loss.item():.4f} "
                        f"| Kin:{kinematic_loss.item():.4f} | Unnorm_Kin:{unnormalized_loss.item():.4f} "
                        f"| Dyn:{dynamic_loss.item():.4f} | KL:{kl_loss.item():.4f} "
                        f"| Sim:{sim_loss.item():.4f}"
                        f"| Beta:{beta:.3f} | TF:{tf_ratio:.2f}"
                    )

            os.makedirs(self.save_path, exist_ok=True)
            save_path = os.path.join(self.save_path, f"vae_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(self.vae.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")
