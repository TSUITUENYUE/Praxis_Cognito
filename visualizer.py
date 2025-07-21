import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch_geometric
from torch.utils.data import DataLoader, SubsetRandomSampler
from Pretrain.dataset import TrajectoryDataset
from Model.vae import IntentionVAE
from Model.agent import Agent
from Pretrain.utils import *

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


class Visualizer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.agent = Agent(**config.agent)
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
        dataset = TrajectoryDataset(processed_path=self.config.trainer.processed_path,
                                    agent=self.agent)

        if subset_fraction < 1.0:
            indices = np.random.choice(len(dataset), int(len(dataset) * subset_fraction), replace=False)
            sampler = SubsetRandomSampler(indices)
            dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size * batch_size_multiplier,
                                          sampler=sampler, num_workers=8)
        else:
            dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size * batch_size_multiplier,
                                    shuffle=False, num_workers=8)

        latents = []
        originals_x = []
        originals_traj = []
        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.config.trainer.device)

        with torch.inference_mode():
            for (x, traj) in dataloader:
                x = x.to(self.config.trainer.device)
                edge_index = edge_index.to(self.config.trainer.device)

                recon_traj, joint_cmd, z, *encoder_outputs = self.model(x, edge_index)

                latents.append(z.cpu().numpy())
                originals_x.append(x.cpu().numpy().reshape(x.shape[0], -1))
                originals_traj.append(traj.cpu().numpy().reshape(traj.shape[0], -1))

        latents = np.concatenate(latents, axis=0)
        originals_x = np.concatenate(originals_x, axis=0)
        originals_traj = np.concatenate(originals_traj, axis=0)
        print(f"Collected {latents.shape[0]} latent samples of dimension {self.latent_dim}")
        print(f"Original x flattened shape: {originals_x.shape}")
        print(f"Original traj flattened shape: {originals_traj.shape}")

        self._visualize_data(latents, title_suffix=f"Latent Space (Prior: {self.prior})", use_pca=use_pca,
                             use_tsne=use_tsne)
        self._visualize_data(originals_traj, title_suffix="Trajectories (traj)", use_pca=use_pca, use_tsne=use_tsne)

        print("\nInterpretation Tips:")
        print(
            "- PCA: Look for explained variance >0.8 for good representation; linear clusters indicate correlated dims.")
        print(
            "- t-SNE: Focus on local clusters (e.g., intent groups); ignore global distances. If blob-like, Gaussian prior fits well.")
        print("- Compare latents vs. originals: If latents are more clustered, VAE is compressing meaningfully.")

    def _visualize_data(self, data: np.ndarray, title_suffix: str, use_pca: bool = True, use_tsne: bool = True):
        if data.shape[1] <= 1:
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=50, density=True, color='white')
            plt.title(f"Histogram of {title_suffix}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.gca().set_facecolor('black')
            plt.gcf().set_facecolor('black')
            plt.show()
            return

        if use_pca:
            pca2d = PCA(n_components=2)
            reduced_2d = pca2d.fit_transform(data)
            explained_var = pca2d.explained_variance_ratio_.sum()
            print(f"PCA 2D explained variance for {title_suffix}: {explained_var:.2f}")
            self._plot_2d_density(reduced_2d, title=f"PCA:Density of {title_suffix}", xlabel="PC1",
                                  ylabel="PC2")

        if use_tsne:
            tsne2d = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            reduced_2d = tsne2d.fit_transform(data)
            print(f"t-SNE 2D completed for {title_suffix} (KL divergence: {tsne2d.kl_divergence_:.4f})")
            self._plot_2d_density(reduced_2d, title=f"t-SNE:Density of {title_suffix}", xlabel="Dim 1",
                                  ylabel="Dim 2")

    def _plot_2d_density(self, reduced_2d: np.ndarray, title: str, xlabel: str, ylabel: str):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Compute KDE
        xy = np.vstack([reduced_2d[:, 0], reduced_2d[:, 1]])
        kde = gaussian_kde(xy)

        # Create grid
        xmin, xmax = reduced_2d[:, 0].min(), reduced_2d[:, 0].max()
        ymin, ymax = reduced_2d[:, 1].min(), reduced_2d[:, 1].max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)

        # Plot with contourf for smooth, continuous density
        cf = ax.contourf(X, Y, Z, cmap='viridis', levels=1000)  # Adjust levels for smoothness
        fig.colorbar(cf, ax=ax, label='Density')

        ax.set_title(title, color='white')
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.set_facecolor('black')
        fig.set_facecolor('black')
        ax.tick_params(axis='both', colors='white')

        # Save the plot
        plot_type = 'PCA' if 'PCA' in title else 'tSNE'
        data_type = 'latent' if 'Latent Space' in title else 'traj'
        save_path = os.path.join(self.config.trainer.save_path, f"{plot_type}_{data_type}.png")
        plt.savefig(save_path, bbox_inches='tight', facecolor='black')

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
    visualizer.visualize_latent_space(subset_fraction=0.05)