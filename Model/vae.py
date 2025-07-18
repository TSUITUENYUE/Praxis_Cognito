import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import GMMEncoder, PoincareEncoder, VanillaEncoder
from .decoder import Decoder
import geoopt
from torch.distributions import MultivariateNormal


class IntentionVAE(nn.Module):
    def __init__(self, prior, node_features, hidden_features, num_layers, rnn_hidden, latent_dim, seq_len, agent, hidden_dim, num_components=128):
        super(IntentionVAE, self).__init__()
        self.prior = prior
        self.num_components = num_components if prior == "GMM" else None
        self.latent_dim = latent_dim
        self.agent = agent
        self.object_dim = agent.object_dim
        self.joint_dim = agent.n_dofs
        self.urdf = agent.urdf

        # Initialize encoder based on prior
        if self.prior == "GMM":
            self.encoder = GMMEncoder(node_features, hidden_features, num_layers, rnn_hidden, latent_dim, num_components)
            # Define the GMM Prior
            prior_mu = torch.randn(num_components, latent_dim) * 5.0  # Spread them out
            self.register_buffer("prior_mu", prior_mu)
            # Prior variance is identity
            self.register_buffer("prior_logvar", torch.zeros(num_components, latent_dim))
            # Prior probability for each component is uniform
            self.register_buffer("prior_pi", torch.full((num_components,), 1.0 / num_components))
        elif self.prior == "Hyperbolic":
            self.encoder = PoincareEncoder(node_features, hidden_features, num_layers, rnn_hidden, latent_dim)
            self.manifold = self.encoder.manifold
            self.device = self.manifold.k.device
            self.register_buffer("prior_mean", torch.zeros(1, latent_dim))
            self.register_buffer("prior_var", torch.ones(1, latent_dim))
        elif self.prior == "Gaussian":
            self.encoder = VanillaEncoder(node_features, hidden_features, num_layers, rnn_hidden, latent_dim)
        else:
            raise ValueError(f"Unsupported prior: {prior}. Choose from 'GMM', 'Hyperbolic', or 'Gaussian'.")

        self.decoder = Decoder(latent_dim, seq_len, self.object_dim, self.joint_dim, self.urdf, hidden_dim)

    def reparameterize_gmm(self, mu, logvar, pi_logits):
        # Get component probabilities
        pi = F.softmax(pi_logits, dim=-1)

        if self.training:
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

    def reparameterize_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def make_covar(self, var):
        if len(var.shape) == 1:
            var = var.unsqueeze(0)  # Now (1, dim)
        assert len(var.shape) == 2, "Assuming diagonal var (batch, dim)"
        covar = torch.diag_embed(var)
        return covar

    def log_prob(self, z, mean_H, var, u=None, v=None):
        if mean_H.dim() == 1:
            mean_H = mean_H.unsqueeze(0).expand(z.shape[0], -1)
        if var.dim() == 1:
            var = var.unsqueeze(0).expand(z.shape[0], -1)
        covar = self.make_covar(var)

        if u is None or v is None:
            u = self.manifold.logmap(mean_H, z)
            origin = self.manifold.origin(u.shape[0], u.shape[1], device=u.device)
            v = self.manifold.transp(mean_H, origin, u)

        r = u.norm(dim=-1, p=2, keepdim=True)
        d = mean_H.shape[-1]

        # normal log prob
        log_prob_normal = MultivariateNormal(torch.zeros_like(covar)[..., 0], covar).log_prob(v)

        # jacobian adjustment
        log_det_jacob = (d - 1) * (r.sinh().log() - r.log())

        # for small r, use Taylor expansion
        small_r_mask = r.abs() < 1e-4  # Use abs() for safety
        if small_r_mask.any():
            r_small = r[small_r_mask]
            # Using a more stable Taylor expansion for (sinh(r)/r)
            log_det_jacob[small_r_mask] = (d - 1) * torch.log1p(r_small ** 2 / 6 + r_small ** 4 / 120)

        # The probability on the manifold is the tangent space probability
        # adjusted for the change in volume (the jacobian).
        return log_prob_normal - log_det_jacob.squeeze(-1)

    def forward(self, x, edge_index):
        if self.prior == "GMM":
            mu, logvar, pi_logits = self.encoder(x, edge_index)
            z = self.reparameterize_gmm(mu, logvar, pi_logits)
            recon_traj, joint_cmd = self.decoder(z)
            return recon_traj, joint_cmd, z, mu, logvar, pi_logits
        elif self.prior == "Hyperbolic":
            mu, logvar = self.encoder(x, edge_index)
            std = F.softplus(logvar)  # Use softplus for stability
            var = std ** 2
            z = self.manifold.wrapped_normal(*mu.shape, mean=mu, std=std)
            recon_traj, joint_cmd = self.decoder(z)
            return recon_traj, joint_cmd, z, mu, var
        elif self.prior == "Gaussian":
            mu, logvar = self.encoder(x, edge_index)
            z = self.reparameterize_gaussian(mu, logvar)
            recon_traj, joint_cmd = self.decoder(z)
            return recon_traj, joint_cmd, z, mu, logvar
        else:
            raise ValueError(f"Unsupported prior in forward: {self.prior}")

    def loss(self, recon_traj, orig_traj, *args, beta):
        # Common reconstruction loss
        recon_loss_pos = F.mse_loss(recon_traj, orig_traj, reduction='mean')
        recon_vel = recon_traj[:, 1:] - recon_traj[:, :-1]
        orig_vel = orig_traj[:, 1:] - orig_traj[:, :-1]
        recon_loss_vel = F.mse_loss(recon_vel, orig_vel)
        total_recon_loss = recon_loss_pos + 0.1 * recon_loss_vel

        if self.prior == "GMM":
            mu, logvar, pi_logits = args
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
        elif self.prior == "Hyperbolic":
            z, mu, var = args
            log_qz = self.log_prob(z, mu, var)
            log_pz = self.log_prob(z, self.prior_mean, self.prior_var)
            kl_loss = (log_qz - log_pz).mean()
        elif self.prior == "Gaussian":
            mu, logvar = args
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            raise ValueError(f"Unsupported prior in loss: {self.prior}")

        # Total VAE loss
        vae_loss = total_recon_loss + beta * kl_loss
        return vae_loss, total_recon_loss, kl_loss