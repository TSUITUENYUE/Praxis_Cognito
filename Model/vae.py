import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import GMMEncoder, PoincareEncoder, VanillaEncoder
from .decoder import Decoder
import geoopt
from torch.distributions import MultivariateNormal
import math




class IntentionVAE(nn.Module):
    def __init__(self, prior, node_features, hidden_features, num_layers, rnn_hidden, latent_dim, seq_len, agent, hidden_dim,obs_dim, fps, num_components=128):
        super(IntentionVAE, self).__init__()
        self.prior = prior
        self.num_components = num_components if prior == "GMM" else None
        self.latent_dim = latent_dim
        self.agent = agent
        self.object_dim = agent.object_dim
        self.joint_dim = agent.n_dofs
        self.urdf = agent.urdf
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_std", torch.ones(3))

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

        self.decoder = Decoder(latent_dim, seq_len, self.object_dim, self.joint_dim, self.agent, hidden_dim, self.pos_mean, self.pos_std, obs_dim, fps)

    def hetero_nll(self, x, mu, log_sigma):
        # x, mu, log_sigma: [B,T,D] in the SAME (normalized) space
        inv_sigma = torch.exp(-log_sigma)
        sq = 0.5 * ((x - mu) * inv_sigma) ** 2
        nll = sq + log_sigma + 0.5 * math.log(2 * math.pi)
        return nll.mean()

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

    def forward(self, x, edge_index, obs_seq=None, q=None, dq=None, tf_ratio: float = 1.0):
        """
        x:           [B, T, N, F]  (graph features)
        edge_index:  PyG edges for a single graph; we internally batch them as before
        teacher_joints: [B,T,DoF] or None
        dq: [B,T,DoF]
        tf_ratio:    float in [0,1], per-step TF probability
        obs_seq:     optional [B,T,obs_dim] if you want RL obs conditioning
        Returns:
          recon_mu:     [B,T,pos_dim]  (normalized)
          joint_traj:   [B,T,DoF]
          actions_seq:  [B,T,DoF]      in [-1,1]
          log_sigma:    [B,T,pos_dim]
          ... latents   (for KL)
        """
        if self.prior == "GMM":
            mu, logvar, pi_logits = self.encoder(x, edge_index)
            z = self.reparameterize_gmm(mu, logvar, pi_logits)
            recon_mu, joint_cmd, actions_seq, log_sigma = self.decoder(
                z, obs_seq=obs_seq, q=q, dq=dq,tf_ratio=tf_ratio
            )
            return recon_mu, joint_cmd, actions_seq, log_sigma, mu, logvar, pi_logits

        elif self.prior == "Hyperbolic":
            mu, logvar = self.encoder(x, edge_index)
            std = F.softplus(logvar)
            z = self.manifold.wrapped_normal(*mu.shape, mean=mu, std=std)
            recon_mu, joint_cmd, actions_seq, log_sigma = self.decoder(
                z, obs_seq=obs_seq, q=q, dq=dq,tf_ratio=tf_ratio
            )
            return recon_mu, joint_cmd, actions_seq, log_sigma, z, mu, std**2

        elif self.prior == "Gaussian":
            mu, logvar = self.encoder(x, edge_index)
            z = self.reparameterize_gaussian(mu, logvar)
            recon_mu, joint_cmd, actions_seq, log_sigma = self.decoder(
                z, obs_seq=obs_seq, q=q, dq=dq,tf_ratio=tf_ratio
            )
            return recon_mu, joint_cmd, actions_seq, log_sigma, z, mu, logvar

    def loss(self, recon_mu, log_sigma, orig_traj, action_seq, act, *args, beta):
        # recon_mu, orig_traj are normalized positions [B,T,D]
        orig_traj = orig_traj.reshape(recon_mu.shape)
        nll_loss = self.hetero_nll(orig_traj, recon_mu, log_sigma)
        kinematic_loss = F.mse_loss(recon_mu, orig_traj)
        dynamic_loss = F.mse_loss(action_seq, act)
        total_recon = kinematic_loss + dynamic_loss
        #total_recon = nll_loss

        # ---- KL based on prior type (unchanged) ----
        if self.prior == "GMM":
            mu, logvar, pi_logits = args
            bs = mu.shape[0]
            pi = F.softmax(pi_logits, dim=-1)
            prior_mu_exp = self.prior_mu.unsqueeze(0).expand(bs, -1, -1)
            prior_logvar_exp = self.prior_logvar.unsqueeze(0).expand(bs, -1, -1)
            kl_gaussians = 0.5 * torch.sum(
                prior_logvar_exp - logvar - 1 +
                (logvar - prior_logvar_exp).exp() +
                (mu - prior_mu_exp).pow(2) / prior_logvar_exp.exp(),
                dim=-1
            )
            kl_categorical = torch.sum(pi * (torch.log(pi + 1e-10) - torch.log(self.prior_pi + 1e-10)), dim=-1)
            kl_loss = torch.sum(pi * kl_gaussians, dim=-1) + kl_categorical
            kl_loss = kl_loss.mean()
        elif self.prior == "Hyperbolic":
            z, mu, var = args
            log_qz = self.log_prob(z, mu, var)
            log_pz = self.log_prob(z, self.prior_mean, self.prior_var)
            kl_loss = (log_qz - log_pz).mean()
        elif self.prior == "Gaussian":
            z, mu, logvar = args
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_per_sample.mean()
        else:
            raise ValueError(f"Unsupported prior in loss: {self.prior}")

        vae_loss = total_recon + beta * kl_loss
        return vae_loss, kinematic_loss, dynamic_loss, kl_loss
