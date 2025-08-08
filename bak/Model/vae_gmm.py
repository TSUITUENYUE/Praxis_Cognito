import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import GMMEncoder
from .decoder import Decoder


def reparameterize_gmm(mu, logvar, pi_logits, training=True):

    # Get component probabilities
    pi = F.softmax(pi_logits, dim=-1)

    if training:
        component_indices = torch.multinomial(pi, 1).squeeze(dim=-1)
    else:
        component_indices = torch.argmax(pi, dim=-1)

    # Select the mu and logvar for the chosen component for each item in the batch
    batch_indices = torch.arange(mu.shape[0], device=mu.device)
    mu_selected = mu[batch_indices, component_indices]
    logvar_selected = logvar[batch_indices, component_indices]

    # Standard reparameterization on the selected component
    std = torch.exp(0.5 * logvar_selected)
    eps = torch.randn_like(std)
    return mu_selected + eps * std


# Full VAE model.
class IntentionVAE(nn.Module):
    def __init__(self, prior, node_features, hidden_features, num_layers, rnn_hidden, latent_dim, seq_len, agent, num_components=128):
        super(IntentionVAE, self).__init__()
        self.prior = prior
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.agent = agent
        joint_dim = self.agent.n_dofs
        urdf = self.agent.urdf
        object_dim = self.agent.object_dim
        # Pass num_components to the Encoder
        self.encoder = GMMEncoder(node_features, hidden_features, num_layers, rnn_hidden, latent_dim, num_components)
        self.decoder = Decoder(latent_dim, seq_len, object_dim, joint_dim, urdf)

        # --- Define the GMM Prior ---
        # Make the prior means non-trainable buffers
        prior_mu = torch.randn(num_components, latent_dim) * 5.0  # Spread them out
        self.register_buffer("prior_mu", prior_mu)
        # Prior variance is identity
        self.register_buffer("prior_logvar", torch.zeros(num_components, latent_dim))
        # Prior probability for each component is uniform
        self.register_buffer("prior_pi", torch.full((num_components,), 1.0 / num_components))

    def forward(self, x, edge_index):
        # Encoder now returns parameters for the GMM
        mu, logvar, pi_logits = self.encoder(x, edge_index)

        # Use the GMM reparameterization trick
        z = reparameterize_gmm(mu, logvar, pi_logits, self.training)

        recon_traj = self.decoder(z)
        return recon_traj, mu, logvar, pi_logits

    def loss(self, recon_traj, orig_traj, mu, logvar, pi_logits, beta):
        # --- Reconstruction Loss (same as before) ---
        recon_loss_pos = F.mse_loss(recon_traj, orig_traj, reduction='mean')
        recon_vel = recon_traj[:, 1:] - recon_traj[:, :-1]
        orig_vel = orig_traj[:, 1:] - orig_traj[:, :-1]
        recon_loss_vel = F.mse_loss(recon_vel, orig_vel)
        total_recon_loss = recon_loss_pos + 0.1 * recon_loss_vel

        # --- NEW: GMM KL Divergence ---
        bs = mu.shape[0]
        pi = F.softmax(pi_logits, dim=-1)  # q(c|x)

        # Expand prior to batch size for calculations
        prior_mu_exp = self.prior_mu.unsqueeze(0).expand(bs, -1, -1)
        prior_logvar_exp = self.prior_logvar.unsqueeze(0).expand(bs, -1, -1)

        # 1. KL divergence between Gaussians: KL(q(z|x,c) || p(z|c))
        # This has shape [bs, k]
        kl_gaussians = 0.5 * torch.sum(
            prior_logvar_exp - logvar - 1 +
            (logvar - prior_logvar_exp).exp() +
            (mu - prior_mu_exp).pow(2) / prior_logvar_exp.exp(),
            dim=-1
        )

        # 2. KL divergence between Categorical distributions: KL(q(c|x) || p(c))
        # This has shape [bs]
        kl_categorical = torch.sum(pi * (torch.log(pi + 1e-10) - torch.log(self.prior_pi + 1e-10)), dim=-1)

        # Combine them: E_{q(c|x)} [ KL(q(z|x,c) || p(z|c)) ] + KL(q(c|x) || p(c))
        kl_loss = torch.sum(pi * kl_gaussians, dim=-1) + kl_categorical

        # Average over the batch
        kl_loss = kl_loss.mean()

        # Total VAE loss
        vae_loss = total_recon_loss + beta * kl_loss
        return vae_loss, total_recon_loss, kl_loss