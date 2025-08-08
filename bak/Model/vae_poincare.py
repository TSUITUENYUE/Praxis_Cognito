import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import PoincareEncoder, VanillaEncoder
from .decoder import Decoder
import geoopt
from torch.distributions import MultivariateNormal

# Full VAE model.
class IntentionVAE(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim, seq_len, object_dim,
                 joint_dim, urdf):
        super(IntentionVAE, self).__init__()
        self.encoder = PoincareEncoder(node_features, hidden_features, num_layers, rnn_hidden, latent_dim)
        self.decoder = Decoder(latent_dim, seq_len, object_dim, joint_dim, urdf)
        self.manifold = self.encoder.manifold
        self.latent_dim = latent_dim
        self.device = self.manifold.k.device
        self.prior_mean = torch.zeros(1, latent_dim, device=self.device)
        self.prior_var = torch.ones(1, latent_dim, device=self.device)

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

        # The incorrect scaling line has been removed.
        # lambda_mu = self.manifold.lambda_x(mean_H, keepdim=True)
        # v = v * lambda_mu  <--- DELETED

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
        mu, logvar = self.encoder(x, edge_index)  # mu is now a point on the manifold
        #logvar = torch.clamp(logvar, min=-6, max=4)
        std = F.softplus(logvar)  # Use softplus for stability
        var = std ** 2

        #print(mu[0], var[0])
        z = self.manifold.wrapped_normal(*mu.shape, mean=mu, std=std)
        recon_traj = self.decoder(z)
        return recon_traj, z, mu, var

    def loss(self, recon_traj, orig_traj, z,mu, var, beta):
        # Reconstruction loss: MSE
        recon_loss = F.mse_loss(recon_traj, orig_traj, reduction='mean')

        recon_vel = recon_traj[:, 1:] - recon_traj[:, :-1]
        orig_vel = orig_traj[:, 1:] - orig_traj[:, :-1]
        recon_loss_vel = F.mse_loss(recon_vel, orig_vel)

        # Combine them (alpha is a hyperparameter to balance the two)
        alpha = 0.1
        recon_loss = recon_loss + alpha * recon_loss_vel
        # KL divergence

        log_qz = self.log_prob(z, mu, var)
        log_pz = self.log_prob(z, self.prior_mean, self.prior_var)
        kl_loss = (log_qz - log_pz).mean()
        # Total VAE loss
        vae_loss = recon_loss + beta * kl_loss
        return vae_loss, recon_loss, kl_loss