import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from .dataset import TrajectoryDataset
from Model.agent import Agent
from Model.vae import IntentionVAE
from .utils import *
import sys
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import geoopt
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_beta(epoch, total_epochs, recon_loss=None, kl_loss=None, strategy='cyclical',
             num_cycles=4, max_beta=1.0, warmup_epochs=20, adaptive_threshold=2.0):
    if strategy == 'warmup':
        if epoch < warmup_epochs:
            return 0.0
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max_beta * progress

    elif strategy == 'cyclical':
        cycle_length = total_epochs // num_cycles
        if cycle_length == 0:
            return max_beta  # Avoid division by zero for small total_epochs
        cycle_progress = (epoch % cycle_length) / cycle_length
        # Use sigmoid for smoother ramp; alternatively, use linear: max_beta * cycle_progress
        sigmoid_progress = 1 / (1 + math.exp(-10 * (cycle_progress - 0.5)))  # Centered sigmoid
        return max_beta * sigmoid_progress

    elif strategy == 'adaptive':
        if recon_loss is None or kl_loss is None:
            raise ValueError("recon_loss and kl_loss are required for adaptive strategy.")
        if kl_loss == 0:  # Avoid division by zero
            return max_beta
        loss_ratio = recon_loss / kl_loss
        if loss_ratio > adaptive_threshold:
            return max(0.0, max_beta * (loss_ratio / adaptive_threshold - 1))  # Decrease beta if recon >> KL
        else:
            return min(max_beta, max_beta * (adaptive_threshold / loss_ratio))  # Increase beta if KL dominates

    else:
        raise ValueError(f"Invalid strategy: {strategy}. Choose from 'warmup', 'cyclical', or 'adaptive'.")

class Trainer:
    def __init__(self, model, config: DictConfig):
        self.load_path = config.load_path
        self.save_path = config.save_path
        self.batch_size = config.batch_size

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
        self.vae = torch.compile(self.vae, dynamic=True).to(self.device)

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

        self.optimizer = optim.Adam(self.vae.parameters(), lr=config.optimizer.lr, fused=True)
        if self.vae_prior == "Hyperbolic":
            self.optimizer = geoopt.optim.RiemannianAdam(self.vae.parameters(), lr=config.optimizer.lr, fused=True)
        num_total_steps = self.num_epochs * (self.episodes // self.batch_size)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_total_steps)
        self.dataset = TrajectoryDataset(processed_path=config.processed_path, source_path=self.load_path, agent=self.agent)
        #self.pos_min = self.dataset.pos_min
        #self.pos_max = self.dataset.pos_max

        self.vae.pos_mean = self.dataset.pos_mean
        self.vae.pos_std = self.dataset.pos_std

    def train(self):
        torch.set_float32_matmul_precision('high')
        print(self.vae.pos_mean, self.vae.pos_std)
        self.vae.train()
        save_interval = max(1, self.num_epochs // 4)
        #scaler = GradScaler()
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=4,
                                shuffle=True,
                                drop_last=True,)

        for epoch in range(self.num_epochs):
            for i, (graph_x, orig_traj) in enumerate(dataloader):
                graph_x = graph_x.to(self.device, non_blocking=True)
                orig_traj = orig_traj.to(self.device, non_blocking=True)

                if self.vae_prior == "Gaussian":
                    recon_traj, _, z, mu, logvar = self.vae(graph_x, self.edge_index)
                    beta = get_beta(epoch=i, total_epochs=self.episodes//self.batch_size, strategy=self.strategy, warmup_epochs=self.warm_up, max_beta=self.max_beta)
                    loss, recon_loss, kl_loss = self.vae.loss(recon_traj, orig_traj, mu, logvar, beta=beta)

                elif self.vae_prior == "GMM":
                    recon_traj,_, z, mu, logvar, pi = self.vae(graph_x, self.edge_index)
                    beta = get_beta(epoch=i, total_epochs=self.episodes//self.batch_size, strategy=self.strategy, warmup_epochs=self.warm_up, max_beta=self.max_beta)
                    loss, recon_loss, kl_loss = self.vae.loss(recon_traj, orig_traj, mu, logvar, pi, beta=beta)

                elif self.vae_prior == "Hyperbolic":
                    recon_traj, _, z, mu, var = self.vae(graph_x, self.edge_index)
                    beta = get_beta(epoch=i, total_epochs=self.episodes//self.batch_size, strategy=self.strategy, warmup_epochs=self.warm_up, max_beta=self.max_beta)
                    loss, recon_loss, kl_loss = self.vae.loss(recon_traj, orig_traj, z,mu,var, beta = beta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(
                    f"Batch {i + 1}/{self.episodes//self.batch_size}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

            if (epoch + 1) % save_interval == 0 or (epoch + 1) == self.num_epochs:
                if not os.path.exists(self.save_path): os.makedirs(self.save_path)
                save_path = self.save_path + f"vae_checkpoint_epoch_{epoch+1}.pth"
                torch.save(self.vae.state_dict(), save_path)
                print(f"Saved model checkpoint to {save_path}")
