import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch_geometric  # Assuming you have torch-geometric installed in your environment
from torch.utils.data import DataLoader, SubsetRandomSampler
# Assuming your dataset class is defined in Pretrain.train or elsewhere; replace with actual import
from Pretrain.dataset import TrajectoryDataset
from Model.vae import IntentionVAE
from Model.agent import Agent
from Pretrain.utils import *

from mpl_toolkits.mplot3d import Axes3D  # For 3D plots


# Paste your IntentionVAE class definition here if not importable
# For brevity, assuming it's importable as from Model.vae import IntentionVAE

class Visualizer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.agent = Agent(**config.agent)  # Assuming Agent is importable
        self.model = IntentionVAE(agent=self.agent, **config.model.vae)
        self.model = torch.compile(self.model).to(self.config.trainer.device)

        checkpoint_path = self.config.trainer.save_path + f"vae_checkpoint_epoch_{self.config.trainer.num_epochs}.pth"
        state_dict = torch.load(checkpoint_path, map_location=self.config.trainer.device)

        self.model.load_state_dict(state_dict)
        self.model.to(self.config.trainer.device)
        self.model.eval()
        self.prior = self.config.model.vae.prior
        self.latent_dim = self.config.model.vae.latent_dim

    def visualize_latent_space(self, subset_fraction=1.0, batch_size_multiplier=2):
        # Load the dataset (assuming it's the test or validation set; adjust as needed)
        # Replace with your actual dataset loading logic, e.g., from generate_dataset or Pretrain.train
        dataset = TrajectoryDataset(processed_path=self.config.trainer.processed_path,
                                    agent=self.agent)  # Example; adjust parameters

        # For speed: Use a random subset if fraction <1
        if subset_fraction < 1.0:
            indices = np.random.choice(len(dataset), int(len(dataset) * subset_fraction), replace=False)
            sampler = SubsetRandomSampler(indices)
            dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size * batch_size_multiplier,
                                    sampler=sampler, num_workers=8,
                                    pin_memory=True)  # Increased workers, pin_memory for GPU
        else:
            dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size * batch_size_multiplier,
                                    shuffle=False, num_workers=8, pin_memory=True)

        latents = []
        originals_x = []  # Collect flattened x
        originals_traj = []  # Collect flattened traj
        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.config.trainer.device)

        with torch.inference_mode():  # Faster than no_grad for inference
            for (x, traj) in dataloader:
                x = x.to(self.config.trainer.device)
                edge_index = edge_index.to(self.config.trainer.device)

                # Run forward to get z (and other outputs, but we only need z)
                recon_traj, joint_cmd, z, *encoder_outputs = self.model(x, edge_index)

                latents.append(z.cpu().numpy())

                # Collect originals (flatten per sample)
                originals_x.append(
                    x.cpu().numpy().reshape(x.shape[0], -1))  # Flatten: (batch, nodes*features) or similar
                originals_traj.append(traj.cpu().numpy().reshape(traj.shape[0], -1))  # Flatten: (batch, time*dims)

        latents = np.concatenate(latents, axis=0)
        originals_x = np.concatenate(originals_x, axis=0)
        originals_traj = np.concatenate(originals_traj, axis=0)
        print(f"Collected {latents.shape[0]} latent samples of dimension {self.latent_dim}")
        print(f"Original x flattened shape: {originals_x.shape}")
        print(f"Original traj flattened shape: {originals_traj.shape}")

        # Visualize latent space (original code, with added 3D PCA)
        if self.latent_dim == 1:
            # Histogram for 1D
            plt.figure(figsize=(10, 6))
            plt.hist(latents.flatten(), bins=50, density=True)
            plt.title(f"Histogram of Latent Space (Prior: {self.prior})")
            plt.xlabel("Latent Value")
            plt.ylabel("Density")
            plt.show()
        elif self.latent_dim == 2:
            # Scatter plot for 2D
            plt.figure(figsize=(10, 6))
            plt.scatter(latents[:, 0], latents[:, 1], alpha=0.5)
            plt.title(f"Scatter Plot of 2D (Prior: {self.prior})")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.show()

            # Optional: 2D histogram for density
            plt.figure(figsize=(10, 6))
            plt.hist2d(latents[:, 0], latents[:, 1], bins=50, cmap='viridis')
            plt.colorbar()
            plt.title(f"2D Histogram of Latent Space (Prior: {self.prior})")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.show()
        else:
            # PCA to 2D for higher dimensions
            pca2d = PCA(n_components=2)
            latents_2d = pca2d.fit_transform(latents)
            explained_variance_2d = pca2d.explained_variance_ratio_.sum()
            print(f"PCA explained variance ratio for 2 components: {explained_variance_2d:.2f}")

            plt.figure(figsize=(10, 6))
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.5)
            plt.title(f"PCA Projection of Latent Space to 2D (Prior: {self.prior})")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.show()

            # Added: 3D PCA for better density sense
            pca3d = PCA(n_components=3)
            latents_3d = pca3d.fit_transform(latents)
            explained_variance_3d = pca3d.explained_variance_ratio_.sum()
            print(f"PCA explained variance ratio for 3 components: {explained_variance_3d:.2f}")

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(latents_3d[:, 0], latents_3d[:, 1], latents_3d[:, 2], alpha=0.3,
                       s=20)  # Smaller s, lower alpha for density
            ax.set_title(f"3D Projection of Latent Space (Prior: {self.prior})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            plt.show()

            # Additional: Per-dimension histograms to check marginal distributions
            fig, axs = plt.subplots(1, min(self.latent_dim, 32), figsize=(32, 5))  # Show up to 5 dimensions
            if self.latent_dim == 1:
                axs = [axs]
            for i in range(min(self.latent_dim, 32)):
                axs[i].hist(latents[:, i], bins=50, density=True)
            axs[i].set_title(f"Dim {i + 1}")
            plt.suptitle(f"Marginal Histograms (Prior: {self.prior})")
            plt.show()

            # Interpretation tip
            print("To determine the likely distribution:")
            print("- If points are clustered around (0,0) in a blob, it might resemble a Gaussian.")
            print("- If multiple clusters appear, it could be a GMM or multimodal.")
            print("- For Hyperbolic prior, points might be distributed in a disk-like manner if 2D.")
            print("- Check histograms for normality (bell-shaped) or other shapes.")

            # New: Visualize original data
            self._visualize_original_data(originals_x, originals_traj)

    def _visualize_original_data(self, originals_x_flat, originals_traj_flat):
        # Helper method for original data vis
        for name, data in [('Input Features (x)', originals_x_flat), ('Trajectories (traj)', originals_traj_flat)]:
            print(f"\nVisualizing distribution for {name} (flattened dim: {data.shape[1]})")

            # PCA to 2D
            pca2d = PCA(n_components=2)
            data_2d = pca2d.fit_transform(data)
            explained_variance_2d = pca2d.explained_variance_ratio_.sum()
            print(f"PCA explained variance for 2 components: {explained_variance_2d:.2f}")

            plt.figure(figsize=(10, 6))
            plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)
            plt.title(f"PCA Projection to 2D for {name}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.show()

            # Added: 3D PCA
            pca3d = PCA(n_components=3)
            data_3d = pca3d.fit_transform(data)
            explained_variance_3d = pca3d.explained_variance_ratio_.sum()
            print(f"PCA explained variance for 3 components: {explained_variance_3d:.2f}")

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], alpha=0.3, s=20)
            ax.set_title(f"3D Projection of {name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            plt.show()

            # Marginal histograms (first 5 dims of flattened)
            fig, axs = plt.subplots(1, 1, figsize=(15, 3))
            for i in range(min(5, data.shape[1])):
                axs[i].hist(data[:, i], bins=50, density=True)
                axs[i].set_title(f"Dim {i + 1}")
            plt.suptitle(f"Marginal Histograms for {name}")
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./conf/go2v.yaml")
    args, unknown_args = parser.parse_known_args()

    config_dir = os.path.dirname(args.config) or "."
    config_name = os.path.basename(args.config).rstrip('.yaml')

    hydra.initialize(version_base=None, config_path=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=unknown_args)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    visualizer = Visualizer(cfg)
    visualizer.visualize_latent_space(subset_fraction=0.2)  # Example: use 20% for speed; set to 1.0 for full