# train.py (drop-in Trainer using the policy decoder outputs)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import os, math

from .dataset import TrajectoryDataset
from .utils import build_edge_index
import geoopt

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
    def __init__(self, model, config):
        self.load_path   = config.load_path
        self.save_path   = config.save_path
        self.batch_size  = config.batch_size
        self.device      = config.device
        self.num_epochs  = config.num_epochs
        self.grad_clip_value = 1.0

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

        self.end_effector_indices = self.agent.end_effector
        self.edge_index = build_edge_index(self.fk_model, self.end_effector_indices, self.device)

        # dataset
        self.dataset = TrajectoryDataset(processed_path=config.processed_path, source_path=self.load_path, agent=self.agent)

        # push dataset normalization into VAE + decoder buffers
        pm = self.dataset.pos_mean.to(self.device); ps = self.dataset.pos_std.to(self.device)
        with torch.no_grad():
            self.vae.pos_mean.copy_(pm); self.vae.pos_std.copy_(ps)
            self.vae.decoder.pos_mean.copy_(pm); self.vae.decoder.pos_std.copy_(ps)

        # optimizer (give policy head a bit more LR)
        base_lr = config.optimizer.lr
        dec = self.vae.decoder

        policy_params = list(dec.policy_head.parameters())
        var_head_params = list(dec.var_head.parameters())

        # compare by id(), not tensor equality
        policy_ids = {id(p) for p in policy_params}
        var_head_ids = {id(p) for p in var_head_params}

        other_params = []
        for n, p in self.vae.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if (pid in policy_ids) or (pid in var_head_ids):
                continue
            other_params.append(p)

        if self.vae_prior == "Hyperbolic":
            # geoopt supports param groups
            self.optimizer = geoopt.optim.RiemannianAdam(
                [
                    {"params": policy_params, "lr": base_lr * 3.0},
                    {"params": var_head_params, "lr": base_lr},
                    {"params": other_params, "lr": base_lr},
                ]
            )
        else:
            self.optimizer = optim.AdamW(
                [
                    {"params": policy_params, "lr": base_lr * 3.0},
                    {"params": var_head_params, "lr": base_lr},
                    {"params": other_params, "lr": base_lr},
                ],
                weight_decay=1e-5,
                fused=True,
            )

        num_total_steps = self.num_epochs * max(1, (self.episodes // self.batch_size))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_total_steps)
        self.scaler = GradScaler()

        # beta schedule cfg
        self.strategy = config.beta_anneal.strategy
        self.warm_up  = config.beta_anneal.warm_up
        self.max_beta = config.beta_anneal.max_beta

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
            for i, (graph_x, teacher_joints) in enumerate(dataloader):
                global_step = epoch * len(dataloader) + i
                graph_x = graph_x.to(self.device)
                teacher_joints = teacher_joints.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    beta = get_beta(global_step, total_steps, strategy=self.strategy,
                                    warmup_epochs=self.warm_up, max_beta=self.max_beta)

                    # teacher forcing ratio: decay to 0 by midway
                    tf_ratio = max(0.0, 1.0 - global_step / (0.5 * total_steps))

                    # --- forward (policy decoder returns actions) ---
                    recon_mu, joint_cmd, actions_seq, log_sigma, *aux = self.vae(
                        graph_x, self.edge_index,
                        teacher_joints=teacher_joints,
                        tf_ratio=tf_ratio,
                        obs_seq=None,
                    )

                    # --- main loss: heteroscedastic NLL on normalized graph positions ---
                    # IMPORTANT: target is graph_x (normalized), not orig_traj
                    loss, recon_loss, kl_loss = self.vae.loss(recon_mu, log_sigma, graph_x, *aux, beta=beta)

                    # --- tiny aux: match joint deltas to encourage motion amplitude ---
                    dq_pred = joint_cmd[:, 1:] - joint_cmd[:, :-1]
                    dq_true = teacher_joints[:, 1:] - teacher_joints[:, :-1]
                    L_delta = F.mse_loss(dq_pred, dq_true)
                    loss = loss + 0.05 * L_delta

                    # --- diag: unnormalized recon MSE in world units ---
                    unnormalized_recon_mu = _denorm_positions(recon_mu, self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_graph_x  = _denorm_positions(graph_x.reshape(recon_mu.shape), self.dataset.pos_mean, self.dataset.pos_std)
                    unnormalized_loss = F.mse_loss(unnormalized_recon_mu, unnormalized_graph_x)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                print(f"Batch {i+1}/{len(dataloader)}, Loss:{loss.item():.4f}, "
                      f"Recon(NLL):{recon_loss.item():.4f}, Unnorm_MSE:{unnormalized_loss.item():.4f}, "
                      f"KL:{kl_loss.item():.4f}, Δq_aux:{L_delta.item():.4f}, "
                      f"Beta:{beta:.3f}, TF:{tf_ratio:.2f}")

            # (optional) save per epoch
            os.makedirs(self.save_path, exist_ok=True)
            save_path = os.path.join(self.save_path, f"vae_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(self.vae.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")
