import torch
import torch.optim as optim
from .dataset import TrajectoryDataset
from .utils import *
import sys
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import geoopt
import math
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard
import torch.nn.functional as F

@torch.no_grad()
def mean_abs_dq(joints):  # joints: [B,T,DoF]
    dq = joints[:, 1:] - joints[:, :-1]                   # [B,T-1,DoF]
    return dq.abs().mean(dim=(0,1,2))                     # scalar

@torch.no_grad()
def amplitude_ratio(joints_pred, joints_true, eps=1e-8):  # both [B,T,DoF]
    dq_p = joints_pred[:, 1:] - joints_pred[:, :-1]       # [B,T-1,DoF]
    dq_t = joints_true[:, 1:] - joints_true[:, :-1]
    num = dq_p.abs().mean(dim=(1,2))                      # [B]
    den = dq_t.abs().mean(dim=(1,2)) + eps                # [B]
    ratio = (num / den).clamp(max=5.0)                    # [B]
    # Return per-sample and batch mean (useful for logging)
    return ratio, ratio.mean()

@torch.no_grad()
def hf_energy_ratio_batch(joints, fps: float, cutoff_hz: float):  # joints: [B,T,DoF]
    # Remove mean over time to avoid DC dominating
    x = joints - joints.mean(dim=1, keepdim=True)          # [B,T,DoF]
    B, T, D = x.shape
    # rFFT over time
    X = torch.fft.rfft(x.float(), dim=1)                   # [B,F,DoF]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(x.device)  # [F] in Hz
    mask = freqs >= cutoff_hz                              # [F]
    # Power above cutoff vs total
    pow_all = (X.abs()**2).sum(dim=(1,2)) + 1e-12          # [B]
    pow_hf  = (X[:, mask, :].abs()**2).sum(dim=(1,2))      # [B]
    return (pow_hf / pow_all).clamp(0.0, 1.0)              # [B]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_beta(epoch, total_epochs, recon_loss=None, kl_loss=None, strategy='cyclical',
             num_cycles=4, max_beta=1.0, warmup_epochs=20, adaptive_threshold=2.0):
    if strategy == 'warmup':
        if epoch < warmup_epochs:
            return 0.0
        else:
            #print(epoch)
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max_beta * progress

    elif strategy == 'cyclical':
        cycle_length = total_epochs // num_cycles
        if cycle_length == 0:
            return max_beta
        cycle_progress = (epoch % cycle_length) / cycle_length
        sigmoid_progress = 1 / (1 + math.exp(-10 * (cycle_progress - 0.5)))
        return max_beta * sigmoid_progress

    elif strategy == 'adaptive':
        if recon_loss is None or kl_loss is None:
            raise ValueError("recon_loss and kl_loss are required for adaptive strategy.")
        if kl_loss == 0:
            return max_beta
        loss_ratio = recon_loss / kl_loss
        if loss_ratio > adaptive_threshold:
            return max(0.0, max_beta * (loss_ratio / adaptive_threshold - 1))
        else:
            return min(max_beta, max_beta * (adaptive_threshold / loss_ratio))

    else:
        raise ValueError(f"Invalid strategy: {strategy}. Choose from 'warmup', 'cyclical', or 'adaptive'.")


class Trainer:
    def __init__(self, model, config):
        self.load_path = config.load_path
        self.save_path = config.save_path
        self.batch_size = config.batch_size
        self.grad_clip_value = 1.0 * 5

        filename = os.path.basename(self.load_path)
        parts = filename.rstrip('.h5').strip().split()
        num_envs = int(parts[0])
        self.episodes = int(parts[1])
        self.max_episode_seconds = int(parts[2])
        self.frame_rate = int(parts[3])
        self.max_episode_len = self.max_episode_seconds * self.frame_rate

        self.device = config.device
        self.num_epochs = config.num_epochs

        # FK and VAE setup
        self.vae = model.to(self.device)
        self.vae_prior = self.vae.prior
        # ✅ Optimized torch.compile mode for static shapes from drop_last=True

        self.vae = torch.compile(self.vae).to(self.device)
        self.agent = self.vae.agent
        self.n_dofs = self.agent.n_dofs
        self.urdf = self.agent.urdf
        self.fk_model = self.agent.fk_model.to(self.device)

        self.num_links = len(self.fk_model.link_names)
        self.num_nodes = self.num_links + 1
        self.position_dim = self.num_links * 3

        self.strategy = config.beta_anneal.strategy
        self.warm_up = config.beta_anneal.warm_up
        self.max_beta = config.beta_anneal.max_beta

        self.end_effector_indices = self.agent.end_effector
        self.edge_index = build_edge_index(self.fk_model, self.end_effector_indices, self.device)

        base_lr = config.optimizer.lr
        dec = self.vae.decoder

        # params for the Δq head only (mlp_delta + delta_scale)
        delta_params = list(dec.mlp_delta.parameters()) + [dec.delta_scale]

        # everything else (no duplicates)
        other_params = []
        for name, p in self.vae.named_parameters():
            if not p.requires_grad:
                continue
            if any(p is dp for dp in delta_params):
                continue
            other_params.append(p)

        # optional: sanity check
        # print(f"delta_params: {sum(p.numel() for p in delta_params)}",
        #       f"other_params: {sum(p.numel() for p in other_params)}")

        if self.vae_prior == "Hyperbolic":
            # geoopt can take param groups too
            self.optimizer = geoopt.optim.RiemannianAdam(
                [
                    {"params": delta_params, "lr": base_lr * 5.0},
                    {"params": other_params, "lr": base_lr},
                ]
            )
        else:
            self.optimizer = optim.AdamW(
                [
                    {"params": delta_params, "lr": base_lr * 5.0},
                    {"params": other_params, "lr": base_lr},
                ],
                weight_decay=1e-5,
                fused=True,
            )
        if self.vae_prior == "Hyperbolic":
            self.optimizer = geoopt.optim.RiemannianAdam(self.vae.parameters(),
                                                         lr=config.optimizer.lr)  # Fused not available

        num_total_steps = self.num_epochs * (self.episodes // self.batch_size)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_total_steps)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, min_lr=1e-7)
        self.dataset = TrajectoryDataset(processed_path=config.processed_path, source_path=self.load_path,
                                         agent=self.agent)

        pm = self.dataset.pos_mean.to(self.device)
        ps = self.dataset.pos_std.to(self.device)
        print(pm, ps)
        # 🔑 Copy into buffers on BOTH VAE and decoder
        with torch.no_grad():
            self.vae.pos_mean.copy_(pm)
            self.vae.pos_std.copy_(ps)
            self.vae.decoder.pos_mean.copy_(pm)
            self.vae.decoder.pos_std.copy_(ps)

            dl = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=2)
            graph_x_b, _, teacher_joints_b = next(iter(dl))  # [B,T,DoF]
            dq = teacher_joints_b[:, 1:] - teacher_joints_b[:, :-1]  # [B,T-1,DoF]
            dq_med = dq.abs().median(dim=1).values.median(dim=0).values  # [DoF]
            dq_init = torch.clamp(dq_med.to(self.device), 0.02, 0.20)  # 0.02–0.20 rad
            self.vae.decoder.delta_scale.data.copy_(dq_init)
        # ✅ Initialize GradScaler for AMP
        self.scaler = GradScaler()

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'runs'))

    def train(self):
        torch.set_float32_matmul_precision('high')
        self.vae.train()
        save_interval = max(1, self.num_epochs // 4)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=True,
                                drop_last=True)

        # ✅ Calculate total steps for correct annealing schedule
        total_steps = self.num_epochs * len(dataloader)
        for epoch in range(self.num_epochs):
            for i, (graph_x, orig_traj, teacher_joints) in enumerate(dataloader):
                global_step = epoch * len(dataloader) + i

                graph_x = graph_x.to(self.device)
                orig_traj = orig_traj.to(self.device)
                teacher_joints = teacher_joints.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                # ----------------------- forward (AMP) -----------------------
                with autocast(enabled=True):
                    beta = get_beta(epoch=global_step, total_epochs=total_steps,
                                    strategy=self.strategy, warmup_epochs=self.warm_up,
                                    max_beta=self.max_beta)

                    # teacher forcing schedule (your current one)
                    tf_ratio = max(0.0, 0.7 - global_step / total_steps)
                    use_teacher = torch.rand(1).item() < tf_ratio

                    recon_mu, joint_cmd, log_sigma, *aux = self.vae(
                        graph_x, self.edge_index,
                        teacher_joints if use_teacher else None
                    )
                    loss, recon_loss, kl_loss = self.vae.loss(recon_mu, log_sigma, orig_traj, *aux, beta=beta)

                    # (your unnormalized MSE for reference)
                    def _denorm_positions(traj_norm, pos_mean, pos_std):
                        B, T, D = traj_norm.shape
                        num_nodes = D // 3
                        x = traj_norm.reshape(B, T, num_nodes, 3)
                        x = x * pos_std[None, None, None, :].to('cuda') + pos_mean[None, None, None, :].to('cuda')
                        return x.reshape(B, T, D)

                    unnormalized_recon_mu = _denorm_positions(recon_mu, self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_graph_x = _denorm_positions(graph_x.reshape(recon_mu.shape), self.dataset.pos_mean,
                                                             self.dataset.pos_std)
                    unnormalized_loss = F.mse_loss(unnormalized_recon_mu, unnormalized_graph_x)

                # --------------------- motion diagnostics ---------------------
                with torch.no_grad():
                    # 1) mean |Δq_pred|
                    dq_mean = mean_abs_dq(joint_cmd)  # scalar

                    # 2) amplitude ratio (per-sample + batch mean)
                    amp_ratio_samples, amp_ratio_mean = amplitude_ratio(joint_cmd, teacher_joints)

                    # 3) HF energy ratios (prediction vs target)
                    fps = float(self.frame_rate)
                    cutoff_hz = 6.0  # tweak 6–8 Hz for walking
                    hf_pred = hf_energy_ratio_batch(joint_cmd, fps, cutoff_hz)  # [B]
                    hf_true = hf_energy_ratio_batch(teacher_joints, fps, cutoff_hz)  # [B]

                    # "walking-only" slice: top 30% by true |Δq|
                    dq_true_per_sample = (teacher_joints[:, 1:] - teacher_joints[:, :-1]).abs().mean(dim=(1, 2))  # [B]
                    if dq_true_per_sample.numel() >= 4:
                        thr = torch.quantile(dq_true_per_sample, 0.70)
                        mask = dq_true_per_sample >= thr
                    else:
                        mask = dq_true_per_sample >= 0.02  # fallback threshold (rad/frame)

                    if mask.any():
                        amp_ratio_walk = amp_ratio_samples[mask].mean()
                        hf_pred_walk = hf_pred[mask].mean()
                        hf_true_walk = hf_true[mask].mean()
                    else:
                        amp_ratio_walk = torch.tensor(float('nan'), device=self.device)
                        hf_pred_walk = torch.tensor(float('nan'), device=self.device)
                        hf_true_walk = torch.tensor(float('nan'), device=self.device)

                    # Optional console peek
                    if global_step % 1 == 0:
                        print(f"[Diag] |Δq|={dq_mean:.4f}  amp={amp_ratio_mean:.3f}  "
                              f"HF(pred/true)={hf_pred.mean():.3f}/{hf_true.mean():.3f}  "
                              f"(walk amp={amp_ratio_walk.item() if mask.any() else float('nan'):.3f})")

                # -------------------- backward & optimize --------------------
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # -------------------------- logs -----------------------------
                print(f"Batch {i + 1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                      f"Unnorm_Recon:{unnormalized_loss.item():.4f}, KL: {kl_loss.item():.4f}, Beta: {beta:.3f}")

                # main losses
                self.writer.add_scalar('Loss/total', loss.item(), global_step)
                self.writer.add_scalar('Loss/recon', recon_loss.item(), global_step)
                self.writer.add_scalar('Loss/kl', kl_loss.item(), global_step)
                self.writer.add_scalar('Loss/unnorm_recon', unnormalized_loss.item(), global_step)
                self.writer.add_scalar('Beta', beta, global_step)
                self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], global_step)

                # diagnostics
                self.writer.add_scalar('diag/mean_abs_dq', dq_mean.item(), global_step)
                self.writer.add_scalar('diag/amplitude_ratio_mean', amp_ratio_mean.item(), global_step)
                self.writer.add_scalar('diag/hf_ratio_pred_mean', hf_pred.mean().item(), global_step)
                self.writer.add_scalar('diag/hf_ratio_true_mean', hf_true.mean().item(), global_step)
                if mask.any():
                    self.writer.add_scalar('diag_walk/amplitude_ratio', amp_ratio_walk.item(), global_step)
                    self.writer.add_scalar('diag_walk/hf_ratio_pred', hf_pred_walk.item(), global_step)
                    self.writer.add_scalar('diag_walk/hf_ratio_true', hf_true_walk.item(), global_step)

        self.writer.close()  # Close TensorBoard writer at the end
