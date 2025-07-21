import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Added for t-SNE
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

    def visualize_latent_space(self, subset_fraction=1.0, batch_size_multiplier=2, use_pca=True, use_tsne=True):
        # Load the dataset (assuming it's the test or validation set; adjust as needed)
        # Replace with your actual dataset loading logic, e.g., from generate_dataset or Pretrain.train
        dataset = TrajectoryDataset(processed_path=self.config.trainer.processed_path,
                                    agent=self.agent)  # Example; adjust parameters

        # For speed: Use a random subset if fraction <1
        if subset_fraction < 1.0:
            indices = np.random.choice(len(dataset), int(len(dataset) * subset_fraction), replace=False)
            sampler = SubsetRandomSampler(indices)
            dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size * batch_size_multiplier,
                                    sampler=sampler, num_workers=8,)  # Increased workers, pin_memory for GPU
        else:
            dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size * batch_size_multiplier,
                                    shuffle=False, num_workers=8,)

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

        # Visualize latent space and originals using refactored helper
        self._visualize_data(latents, title_suffix=f"Latent Space (Prior: {self.prior})", use_pca=use_pca, use_tsne=use_tsne)
        self._visualize_data(originals_x, title_suffix="Input Features (x)", use_pca=use_pca, use_tsne=use_tsne)
        self._visualize_data(originals_traj, title_suffix="Trajectories (traj)", use_pca=use_pca, use_tsne=use_tsne)

        # Interpretation tips
        print("\nInterpretation Tips:")
        print("- PCA: Look for explained variance >0.8 for good representation; linear clusters indicate correlated dims.")
        print("- t-SNE: Focus on local clusters (e.g., intent groups); ignore global distances. If blob-like, Gaussian prior fits well.")
        print("- Compare latents vs. originals: If latents are more clustered, VAE is compressing meaningfully.")

    def _visualize_data(self, data: np.ndarray, title_suffix: str, use_pca: bool = True, use_tsne: bool = True):
        """Helper for PCA/t-SNE vis in 2D/3D with density plots."""
        if data.shape[1] <= 1:
            # Histogram for 1D (no reduction needed)
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=50, density=True)
            plt.title(f"Histogram of {title_suffix}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.show()
            return

        # 2D Reduction (PCA and/or t-SNE)
        if use_pca:
            pca2d = PCA(n_components=2)
            reduced_2d = pca2d.fit_transform(data)
            explained_var = pca2d.explained_variance_ratio_.sum()
            print(f"PCA 2D explained variance for {title_suffix}: {explained_var:.2f}")
            self._plot_2d_density(reduced_2d, title=f"PCA 2D Hexbin Density of {title_suffix}", xlabel="PC1", ylabel="PC2")

        if use_tsne:
            tsne2d = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            reduced_2d = tsne2d.fit_transform(data)
            print(f"t-SNE 2D completed for {title_suffix} (KL divergence: {tsne2d.kl_divergence_:.4f})")
            self._plot_2d_density(reduced_2d, title=f"t-SNE 2D Hexbin Density of {title_suffix}", xlabel="Dim 1", ylabel="Dim 2")

        # 3D Reduction (if dim >2)
        if data.shape[1] > 2:
            if use_pca:
                pca3d = PCA(n_components=3)
                reduced_3d = pca3d.fit_transform(data)
                explained_var = pca3d.explained_variance_ratio_.sum()
                print(f"PCA 3D explained variance for {title_suffix}: {explained_var:.2f}")
                self._plot_3d_scatter(reduced_3d, title=f"PCA 3D Scatter of {title_suffix}", xlabel="PC1", ylabel="PC2", zlabel="PC3")

            if use_tsne:
                tsne3d = TSNE(n_components=3, perplexity=30, random_state=42, n_jobs=-1)
                reduced_3d = tsne3d.fit_transform(data)
                print(f"t-SNE 3D completed for {title_suffix} (KL divergence: {tsne3d.kl_divergence_:.4f})")
                self._plot_3d_scatter(reduced_3d, title=f"t-SNE 3D Scatter of {title_suffix}", xlabel="Dim 1", ylabel="Dim 2", zlabel="Dim 3")

    def _plot_2d_density(self, reduced_2d: np.ndarray, title: str, xlabel: str, ylabel: str):
        """Hexbin density plot for 2D."""
        plt.figure(figsize=(10, 6))
        plt.hexbin(reduced_2d[:, 0], reduced_2d[:, 1], gridsize=50, cmap='viridis', mincnt=1)
        plt.colorbar(label='Density')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def _plot_3d_scatter(self, reduced_3d: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str):
        """3D scatter with color by third dim as density proxy."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], c=reduced_3d[:, 2], cmap='viridis', alpha=0.3, s=20)
        fig.colorbar(scatter, ax=ax, label='Third Dim (Density Proxy)')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./conf/go2.yaml")
    args, unknown_args = parser.parse_known_args()

    config_dir = os.path.dirname(args.config) or "."
    config_name = os.path.basename(args.config).rstrip('.yaml')

    hydra.initialize(version_base=None, config_path=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=unknown_args)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    visualizer = Visualizer(cfg)
    visualizer.visualize_latent_space(subset_fraction=0.05)  # Toggle use_pca=False or use_tsne=False if needed