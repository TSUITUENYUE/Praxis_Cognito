import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from rsl_rl.modules import EmpiricalNormalization  # kept for compatibility if you want it later

from .dataset import TrajectoryDataset
from .utils import masked_quat_geodesic,masked_mae
from Model.sd import SurrogateDynamics
import torch



class Trainer:
    """
    Pretrain ONLY the surrogate dynamics with supervised 1-step losses.
    We do NOT use the VAE in this file; we only grab `model.agent` (for FK, radii, gains)
    and `model.surrogate` (the exact module we want to pretrain).
    """

    def __init__(self, model, rl_config, config):
        # ---- io / meta ----
        self.load_path   = config.load_path
        self.save_path   = config.save_path
        self.device      = config.device
        self.batch_size  = config.batch_size
        self.num_epochs  = config.num_epochs
        self.grad_clip_value = 1.0

        # Parse dataset filename metadata (same as before)
        filename = os.path.basename(self.load_path)
        parts = filename.rstrip('.h5').strip().split()
        num_envs = int(parts[0]); self.episodes = int(parts[1])
        self.max_episode_seconds = int(parts[2]); self.frame_rate = int(parts[3])
        self.max_episode_len = self.max_episode_seconds * self.frame_rate

        # ---- agent & surrogate (only things we reuse from the given model) ----
        self.agent = model.agent.to(self.device)                   # FK, radii, init angles, gains
        self.sd = model.surrogate.to(self.device)                  # the SurrogateDynamics to pretrain

        # Freeze everything except the surrogate
        for p in model.parameters():
            p.requires_grad_(False)
        for p in self.sd.parameters():
            p.requires_grad_(True)

        # Timing for inner control ticks
        self.dt = 1.0 / float(self.frame_rate)
        self.control_dt = rl_config.env.dt
        self.K = max(1, int(round(self.dt / self.control_dt)))

        # Gains, scales, limits we need to reproduce PD torques used during data collection
        self.kp = rl_config.env.kp
        self.kd = rl_config.env.kd
        self.default_dof_pos = self.agent.init_angles.to(self.device)
        self.action_scale = rl_config.env["action_scale"]
        self.clip_actions = rl_config.env["clip_actions"]
        self.simulate_action_latency = rl_config.env.get("simulate_action_latency", False)

        # Dataset & normalizer (the model-free trainer does not need obs normalization)
        self.dataset = TrajectoryDataset(
            processed_path=config.processed_path,
            source_path=self.load_path,
            agent=self.agent
        )

        # Optimizer/scheduler ONLY over surrogate parameters
        base_lr = config.optimizer.lr
        self.opt = optim.Adam(self.sd.parameters(), lr=base_lr, weight_decay=0, fused=True)
        self.steps_per_epoch = max(1, (self.episodes // self.batch_size))
        self.num_total_steps = self.num_epochs * self.steps_per_epoch
        self.sched = CosineAnnealingLR(self.opt, T_max=self.num_total_steps)

        self.scaler = GradScaler()

        # State loss weights (same keys you already use)
        self.w_q  = config.sim.w_q
        self.w_dq = config.sim.w_dq
        self.w_p  = config.sim.w_p
        self.w_dp = config.sim.w_dp
        self.w_w  = config.sim.w_w
        self.w_dw = config.sim.w_dw
        self.w_u  = config.sim.w_u
        self.w_du = config.sim.w_du
        self.lambda_sim = config.sim.lambda_sim

    # ---------- helpers ----------
    def _pd_torque(self, q, dq, a):
        """
        Genesis-style PD:
          tau = Kp*(q_des - q) + Kd*(0 - dq)
        where q_des = clamp(a,[-clip,clip]) * action_scale + default_dof_pos
        Shapes: q,dq,a: [B,d]
        """
        a_clamped = torch.clamp(a, -self.clip_actions, self.clip_actions)
        q_des = a_clamped * self.action_scale + self.default_dof_pos
        # broadcast-safe per-DOF gains
        B, d = q.shape
        kp = torch.as_tensor(self.kp, device=q.device, dtype=q.dtype)
        kd = torch.as_tensor(self.kd, device=q.device, dtype=q.dtype)
        if kp.ndim == 0: kp = kp.repeat(d)
        if kd.ndim == 0: kd = kd.repeat(d)
        kp = kp.view(1, d).expand(B, d)
        kd = kd.view(1, d).expand(B, d)
        tau = kp * (q_des - q) - kd * dq
        return tau

    @staticmethod
    def _masked_l1(pred, target, mask):
        # pred/target: [B,T,D], mask: [B,T,1]
        per_t = (pred - target).abs().mean(dim=-1)      # [B,T]
        denom = mask.sum().clamp_min(1.0)
        return (per_t * mask.squeeze(-1)).sum() / denom

    def _one_step_predict(self, q_t, dq_t, p_t, dp_t, w_t, dw_t, u_t, du_t, a_t, last_a):
        """
        Integrate K inner control ticks starting from GT state at time t.
        Returns predicted state at t+1 and the action actually executed on first tick (for latency).
        """
        # local copies (no in-place)
        q_k, dq_k = q_t, dq_t
        p_k, dp_k = p_t, dp_t
        w_k, dw_k = w_t, dw_t
        u_k, du_k = u_t, du_t

        # inner ticks
        for k in range(self.K):
            exec_a = last_a if (self.simulate_action_latency and k == 0) else a_t
            tau_pd = self._pd_torque(q_k, dq_k, exec_a)
            q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k = self.sd(
                q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k, tau_pd
            )
        # the command that will be treated as "last" for the next outer step
        next_last_a = a_t
        return (q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k), next_last_a

    # ---------- training ----------
    def train(self):
        self.sd.train()

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

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        global_step = 0

        for epoch in range(self.num_epochs):
            for batch in dataloader:
                # unpack & move
                x, q, dq, p, dp, w, dw, u, du, dv, obs, act, mask = batch
                q = q.to(self.device, non_blocking=True)      # [B,T,d]
                dq = dq.to(self.device, non_blocking=True)
                p = p.to(self.device, non_blocking=True)      # [B,T,3]
                dp = dp.to(self.device, non_blocking=True)
                w = w.to(self.device, non_blocking=True)
                dw = dw.to(self.device, non_blocking=True)
                u = u.to(self.device, non_blocking=True)      # [B,T,3]
                du = du.to(self.device, non_blocking=True)
                act = act.to(self.device, non_blocking=True)  # [B,T,d]
                mask = mask.to(self.device, non_blocking=True)[:, :, None]  # [B,T,1]

                # Assume you already moved batch tensors to device and have:
                # q, dq, p, dp, w, dw, u, du, act, mask  with shapes [B, T, *]
                B, T, d = q.shape
                K = self.K  # inner control ticks
                last_actions = torch.zeros(B, d, device=self.device, dtype=q.dtype)

                # storage for predictions over T-1 steps
                q_pred_list, dq_pred_list = [], []
                p_pred_list, dp_pred_list = [], []
                w_pred_list, dw_pred_list = [], []
                u_pred_list, du_pred_list = [], []

                for t in range(T - 1):
                    q_t, dq_t = q[:, t, :], dq[:, t, :]
                    p_t, dp_t = p[:, t, :], dp[:, t, :]
                    w_t, dw_t = w[:, t, :], dw[:, t, :]
                    u_t, du_t = u[:, t, :], du[:, t, :]
                    a_t = act[:, t, :]

                    # K inner ticks (teacher forcing from GT state at time t)
                    q_k, dq_k = q_t, dq_t
                    p_k, dp_k = p_t, dp_t
                    w_k, dw_k = w_t, dw_t
                    u_k, du_k = u_t, du_t
                    for k in range(K):
                        exec_a = last_actions if (self.simulate_action_latency and k == 0) else a_t
                        tau_pd = self._pd_torque(q_k, dq_k, exec_a)
                        q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k = self.sd(
                            q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k, tau_pd
                        )

                    # cache predictions for step t -> t+1
                    q_pred_list.append(q_k.unsqueeze(1))
                    dq_pred_list.append(dq_k.unsqueeze(1))
                    p_pred_list.append(p_k.unsqueeze(1))
                    dp_pred_list.append(dp_k.unsqueeze(1))
                    w_pred_list.append(w_k.unsqueeze(1))
                    dw_pred_list.append(dw_k.unsqueeze(1))
                    u_pred_list.append(u_k.unsqueeze(1))
                    du_pred_list.append(du_k.unsqueeze(1))

                    last_actions = a_t  # for latency on next outer step

                # stack to [B, T-1, *]
                q_pred = torch.cat(q_pred_list, dim=1)
                dq_pred = torch.cat(dq_pred_list, dim=1)
                p_pred = torch.cat(p_pred_list, dim=1)
                dp_pred = torch.cat(dp_pred_list, dim=1)
                w_pred = torch.cat(w_pred_list, dim=1)
                dw_pred = torch.cat(dw_pred_list, dim=1)
                u_pred = torch.cat(u_pred_list, dim=1)
                du_pred = torch.cat(du_pred_list, dim=1)

                # slice GT and mask to t+1..T-1
                mask_next = mask[:, 1:, :]  # [B, T-1, 1]
                q_gt_n, dq_gt_n = q[:, 1:, :], dq[:, 1:, :]
                p_gt_n, dp_gt_n = p[:, 1:, :], dp[:, 1:, :]
                w_gt_n, dw_gt_n = w[:, 1:, :], dw[:, 1:, :]
                u_gt_n, du_gt_n = u[:, 1:, :], du[:, 1:, :]

                # masked MAE + masked quat geodesic (sequence versions)
                q_loss = masked_mae(q_pred, q_gt_n, mask_next)
                dq_loss = masked_mae(dq_pred, dq_gt_n, mask_next)
                p_loss = masked_mae(p_pred, p_gt_n, mask_next)
                dp_loss = masked_mae(dp_pred, dp_gt_n, mask_next)
                w_loss = masked_quat_geodesic(w_pred, w_gt_n, mask_next)
                dw_loss = masked_mae(dw_pred, dw_gt_n, mask_next)
                u_loss = masked_mae(u_pred, u_gt_n, mask_next)
                du_loss = masked_mae(du_pred, du_gt_n, mask_next)

                state_loss = (
                        self.w_q * q_loss + self.w_dq * dq_loss +
                        self.w_p * p_loss + self.w_dp * dp_loss +
                        self.w_w * w_loss + self.w_dw * dw_loss +
                        self.w_u * u_loss + self.w_du * du_loss
                )
                total_loss = self.lambda_sim * state_loss

                # --- backward / step ---
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.sd.parameters(), self.grad_clip_value)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()

                print(f"state_loss: {state_loss.item():.6f}  total_loss: {total_loss.item():.6f}")

                global_step += 1

            # save surrogate checkpoint only
            os.makedirs(self.save_path, exist_ok=True)
            save_path = os.path.join(self.save_path, f"surrogate_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(self.sd.state_dict(), save_path)
            print(f"Saved surrogate checkpoint to {save_path}")
