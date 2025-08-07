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
        self.grad_clip_value = 1.0

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

        self.optimizer = optim.AdamW(self.vae.parameters(), lr=config.optimizer.lr, weight_decay=1e-5, fused=True)
        if self.vae_prior == "Hyperbolic":
            self.optimizer = geoopt.optim.RiemannianAdam(self.vae.parameters(),
                                                         lr=config.optimizer.lr)  # Fused not available

        num_total_steps = self.num_epochs * (self.episodes // self.batch_size)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_total_steps)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, min_lr=1e-7)
        self.dataset = TrajectoryDataset(processed_path=config.processed_path, source_path=self.load_path,
                                         agent=self.agent)

        self.vae.pos_mean = self.dataset.pos_mean
        self.vae.pos_std = self.dataset.pos_std

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
            for i, (graph_x, orig_traj,teacher_joints) in enumerate(dataloader):
                # ✅ Calculate a global step for the schedulers
                global_step = epoch * len(dataloader) + i

                graph_x = graph_x.to(self.device, non_blocking=True)
                orig_traj = orig_traj.to(self.device, non_blocking=True)
                teacher_joints = teacher_joints.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                # ✅ Wrap forward pass in autocast for AMP
                with autocast(enabled=True):
                    # --- Common operations ---
                    outputs = self.vae(graph_x, self.edge_index, teacher_joints)
                    recon_traj = outputs[0]
                    beta = get_beta(epoch=global_step, total_epochs=total_steps,
                                    strategy=self.strategy, warmup_epochs=self.warm_up,
                                    max_beta=self.max_beta)
                    # --- Specific loss calculation ---
                    if self.vae_prior == "Gaussian":
                        mu, logvar = outputs[3], outputs[4]
                        loss, recon_loss, kl_loss = self.vae.loss(recon_traj, orig_traj, mu, logvar, beta=beta)
                    elif self.vae_prior == "GMM":
                        mu, logvar, pi = outputs[3], outputs[4], outputs[5]
                        loss, recon_loss, kl_loss = self.vae.loss(recon_traj, orig_traj, mu, logvar, pi, beta=beta)
                    elif self.vae_prior == "Hyperbolic":
                        z, mu, var = outputs[2], outputs[3], outputs[4]
                        loss, recon_loss, kl_loss = self.vae.loss(recon_traj, orig_traj, z, mu, var, beta=beta)

                # ✅ Scale the loss and perform backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()
                print(
                    f"Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, Beta: {beta:.3f}")

                # Log to TensorBoard per batch
                self.writer.add_scalar('Loss/total', loss.item(), global_step)
                self.writer.add_scalar('Loss/recon', recon_loss.item(), global_step)
                self.writer.add_scalar('Loss/kl', kl_loss.item(), global_step)
                self.writer.add_scalar('Beta', beta, global_step)
                self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], global_step)

            if (epoch + 1) % save_interval == 0 or (epoch + 1) == self.num_epochs:
                if not os.path.exists(self.save_path): os.makedirs(self.save_path)
                save_path = self.save_path + f"vae_checkpoint_epoch_{epoch + 1}.pth"
                torch.save(self.vae.state_dict(), save_path)
                print(f"Saved model checkpoint to {save_path}")

        self.writer.close()  # Close TensorBoard writer at the end
