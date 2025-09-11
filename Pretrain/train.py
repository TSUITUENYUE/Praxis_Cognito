import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import os, math
import genesis as gs
from rsl_rl.modules import EmpiricalNormalization

from .go2_env_icm import Go2Env
from .dataset import TrajectoryDataset
from .utils import build_edge_index, get_beta, masked_quat_geodesic

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
        # Enable cudnn autotuner for conv-like ops
        torch.backends.cudnn.benchmark = True
        # Compile model for kernel fusion where available
        #self.vae = torch.compile(model.to(self.device), mode="reduce-overhead")
        self.vae = model.to(self.device)
        #self.vae.encoder = torch.compile(self.vae.encoder,mode="reduce-overhead")
        #self.vae.decoder = torch.compile(self.vae.decoder,mode="reduce-overhead")
        #self.vae.surrogate = torch.compile(self.vae.surrogate,mode="reduce-overhead")
        self.agent = self.vae.agent
        self.fk_model = self.agent.fk_model.to(self.device)

        '''
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
        '''
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
        # --- Two optimizers: G (everything except D) and D (discriminator only) ---
        self.base_lr = config.optimizer.lr
        self.scaler = GradScaler()

        # Split params cleanly
        disc_params = list(self.vae.disc_post.parameters())
        disc_ids = {id(p) for p in disc_params}
        gen_params = [p for p in self.vae.parameters() if id(p) not in disc_ids]

        # Adam (fused=True if available)
        self.opt_G = optim.Adam(gen_params, lr=self.base_lr, weight_decay=0, fused=True)
        self.opt_D = optim.Adam(self.vae.disc_post.parameters(), lr=self.base_lr, weight_decay=0, fused=True)

        # Cosine schedulers
        self.steps_per_epoch = max(1, (self.episodes // self.batch_size))
        self.num_total_steps = self.num_epochs * self.steps_per_epoch
        self.sched_G = CosineAnnealingLR(self.opt_G, T_max=self.num_total_steps)
        self.sched_D = CosineAnnealingLR(self.opt_D, T_max=self.num_total_steps)

        # weights
        self.strategy = config.beta_anneal.strategy
        self.warm_up  = config.beta_anneal.warm_up
        self.max_beta = config.beta_anneal.max_beta

        self.lambda_kinematic = config.kino.lambda_kinematic
        self.lambda_dynamic   = config.kino.lambda_dynamic

        self.lambda_sim = config.sim.lambda_sim
        self.w_q  = config.sim.w_q
        self.w_dq = config.sim.w_dq
        self.w_p = config.sim.w_p
        self.w_dp = config.sim.w_dp
        self.w_w = config.sim.w_w
        self.w_dw = config.sim.w_dw
        self.w_u = config.sim.w_u
        self.w_du = config.sim.w_du
        self.w_obs= config.sim.w_obs

    # ---------- helpers ----------
    @staticmethod
    def masked_mse(pred, target, mask):
        # pred/target: [B,T,D], mask: [B,T,1]
        per_t = (torch.abs(pred - target)).mean(dim=-1)  # [B,T]
        #per_t = ((pred - target)**2).mean(dim=-1)  # [B,T]
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

        num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        def _denorm_positions(traj_norm, pos_mean_d, pos_std_d):
            B, T, D = traj_norm.shape
            num_nodes = D // 3
            x = traj_norm.reshape(B, T, num_nodes, 3)
            x = x * pos_std_d[None, None, None, :] + pos_mean_d[None, None, None, :]
            return x.reshape(B, T, D)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        global_step = 0

        for epoch in range(self.num_epochs):
            for i, (x, q, dq, p, dp, dw, u, du, dv, obs, act, mask) in enumerate(dataloader):
                x = x.to(self.device, non_blocking=True)
                q_gt = q.to(self.device, non_blocking=True)
                dq_gt = dq.to(self.device, non_blocking=True)
                p_gt = p.to(self.device, non_blocking=True)
                dp_gt = dp.to(self.device, non_blocking=True)
                dw_gt = dw.to(self.device, non_blocking=True)
                u_gt = u.to(self.device, non_blocking=True)
                du_gt = du.to(self.device, non_blocking=True)
                dv_gt = dv.to(self.device, non_blocking=True)
                obs_gt = obs.to(self.device, non_blocking=True)
                act_gt = act.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)[:, :, None]

                beta = get_beta(global_step, self.num_total_steps, strategy=self.strategy,
                                warmup_epochs=self.warm_up, max_beta=self.max_beta)

                # ============================================================
                # D-STEP: update discriminator on detached G outputs
                # ============================================================
                # 1) Cheap forward for G outputs (no G graph)
                with torch.no_grad(), autocast(enabled=True, dtype=amp_dtype):
                    out_D = self.vae(
                        x, self.edge_index, mask,
                        self.obs_normalizer,
                        obs_seq=obs_gt,
                        q=q_gt, dq=dq_gt, p=p_gt, dp=dp_gt, dw=dw_gt,
                        u=u_gt, du=du_gt, dv=dv_gt,
                    )
                    x_recon_D = out_D["traj"]["graph"]
                    log_sigma_D = out_D["act"]["log_sigma"]
                    ptr_D = out_D["traj"]["ptr"]
                    aux_D = out_D.get("aux", [])
                    act_seq_D = out_D["act"]["act"]
                    mu_seq_D = out_D["act"]["mu"]
                    log_std_D = out_D["act"]["log_std"]

                # 2) Compute D loss with gradients only for D
                self.opt_D.zero_grad(set_to_none=True)
                with autocast(enabled=True, dtype=amp_dtype):
                    _, _, _, _, amp_D_loss = self.vae.loss(
                        x_recon_D, log_sigma_D, x,
                        act_gt, act_seq_D,
                        mu_seq_D, log_std_D,
                        ptr_D, mask, *aux_D,
                        beta=beta,
                        lambda_kinematic=self.lambda_kinematic,
                    )

                if amp_D_loss.requires_grad:
                    self.scaler.scale(amp_D_loss).backward()
                    self.scaler.unscale_(self.opt_D)
                    nn.utils.clip_grad_norm_(self.vae.disc_post.parameters(), self.grad_clip_value)
                    self.scaler.step(self.opt_D)
                    # single scaler for both steps; update after G-step to be safe
                    self.sched_D.step()

                # ============================================================
                # G-STEP: update encoder/decoder/surrogate on full loss
                # ============================================================
                self.opt_G.zero_grad(set_to_none=True)
                with autocast(enabled=True, dtype=amp_dtype):

                    out = self.vae(
                        x, self.edge_index, mask,
                        self.obs_normalizer,
                        obs_seq=obs_gt,
                        q=q_gt, dq=dq_gt, p=p_gt, dp=dp_gt, dw=dw_gt,
                        u=u_gt, du=du_gt, dv=dv_gt,
                    )

                    # ---- trajectory
                    x_recon = out["traj"]["graph"]
                    ptr = out["traj"]["ptr"]

                    # ---- state
                    state = out["state"]
                    q_seq = state.get("q")
                    dq_seq = state.get("dq")
                    w_seq = state.get("w")
                    dw_seq = state.get("dw")
                    p_seq = state.get("p")
                    dp_seq = state.get("dp")

                    # ---- actions
                    act_pack = out["act"]
                    act_seq = act_pack.get("act")
                    mu_seq = act_pack.get("mu")
                    log_std_seq = act_pack.get("log_std")
                    log_sigma = act_pack.get("log_sigma")

                    # ---- obs
                    obs_seq = out["obs"]["obs"]
                    u_seq = obs_seq[:, :, -6:-3]
                    du_seq = obs_seq[:, :, -3:]

                    # ---- aux latents/posterior
                    aux = out.get("aux", [])



                    # VAE loss (pre-MAE + action-reg + post-AMP(G) + KL)
                    loss, pre_pose_mae, post_amp_G,  kl_loss, _ = self.vae.loss(
                        x_recon, log_sigma, x,
                        act_gt, act_seq,
                        mu_seq, log_std_seq,

                        ptr, mask, *aux,
                        beta=beta,
                        lambda_kinematic=self.lambda_kinematic,
                    )

                    # State-space losses
                    q_loss = self.masked_mse(q_seq, q_gt, mask)
                    dq_loss = self.masked_mse(dq_seq, dq_gt, mask)
                    p_loss = self.masked_mse(p_seq, p_gt, mask)
                    dp_loss = self.masked_mse(dp_seq, dp_gt, mask)
                    dw_loss = self.masked_mse(dw_seq, dw_gt, mask)
                    u_loss = self.masked_mse(u_seq, u_gt, mask)
                    du_loss = self.masked_mse(du_seq, du_gt, mask)

                    state_loss = (
                            self.w_q * q_loss +
                            self.w_dq * dq_loss +
                            self.w_p * p_loss +
                            self.w_dp * dp_loss +
                            self.w_dw * dw_loss +
                            self.w_u * u_loss +
                            self.w_du * du_loss
                    )

                    sim_loss = state_loss
                    total_loss = loss + self.lambda_sim * sim_loss

                # Backprop THROUGH total_loss so surrogate learns from sim_loss
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.opt_G)
                nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)
                self.scaler.step(self.opt_G)
                self.scaler.update()
                self.sched_G.step()

                # ------------- logging -------------
                log_every = 1
                grad_log_every = 10
                if (global_step % log_every == 0) or (i == len(dataloader) - 1):
                    with torch.no_grad():
                        unnormalized_recon = _denorm_positions(x_recon.detach(), self.pos_mean_d, self.pos_std_d)
                        unnormalized_gt = _denorm_positions(x.reshape(x_recon.shape).detach(), self.pos_mean_d,
                                                            self.pos_std_d)
                        unnorm_obj = self.masked_mse(
                            unnormalized_recon[:, :, -3:], unnormalized_gt[:, :, -3:], mask
                        ).item()
                        unnorm_kino = self.masked_mse(
                            unnormalized_recon[:, :, :-3], unnormalized_gt[:, :, :-3], mask
                        ).item()

                    print(
                        f"Batch {i + 1}/{len(dataloader)} "
                        f"| Total:{total_loss.item():.4f}  VAE:{loss.item():.4f} "
                        f"| PreMAE:{pre_pose_mae.item():.4f}  AMP_G:{post_amp_G.item():.4f} "
                        f"| Unnorm_Kin:{unnorm_kino:.4f}  Unnorm_Obj:{unnorm_obj:.4f} "
                        f"| KL:{kl_loss.item():.4f} "
                        f"| Sim:{sim_loss.item():.4f}  Beta:{beta:.3f}"
                    )
                    '''
                    if (global_step % grad_log_every == 0) or (i == len(dataloader) - 1):
                        print(self.w_q * q_loss.item(),
                              self.w_dq * dq_loss.item(),
                              self.w_p * p_loss.item(),
                              self.w_dp * dp_loss.item(),
                              self.w_dw * dw_loss.item(),
                              self.w_u * u_loss.item(),
                              self.w_du * du_loss.item())
                              
                    '''

                global_step += 1

            os.makedirs(self.save_path, exist_ok=True)
            save_path = os.path.join(self.save_path, f"vae_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(self.vae.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")
