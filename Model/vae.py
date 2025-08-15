import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GMMEncoder, VanillaEncoder
from .decoder import Decoder
from .sd import SurrogateDynamics

class IntentionVAE(nn.Module):
    def __init__(
        self,
        prior: str,
        node_features: int,
        hidden_features: int,
        num_layers: int,
        rnn_hidden: int,
        latent_dim: int,
        seq_len: int,
        agent,
        hidden_dim: int,
        obs_dim: int,
        fps: int,
        num_components: int = 128,
    ):
        super().__init__()
        assert prior in {"Gaussian", "GMM"}, "Supported priors: 'Gaussian', 'GMM'"
        self.prior = prior
        self.num_components = num_components if prior == "GMM" else None
        self.latent_dim = latent_dim

        self.agent = agent
        self.object_dim = agent.object_dim
        self.joint_dim = agent.n_dofs
        self.urdf = agent.urdf

        # NOTE: replace these with dataset stats (pos_mean/std) before training.
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_std", torch.ones(3))

        # --- Encoder ---
        if self.prior == "GMM":
            self.encoder = GMMEncoder(
                node_features, hidden_features, num_layers, rnn_hidden, latent_dim, num_components
            )
            # GMM prior (uniform mixing, diagonal unit variance)
            prior_mu = torch.randn(num_components, latent_dim) * 5.0
            self.register_buffer("prior_mu", prior_mu)
            self.register_buffer("prior_logvar", torch.zeros(num_components, latent_dim))
            self.register_buffer("prior_pi", torch.full((num_components,), 1.0 / num_components))
        else:  # Gaussian
            self.encoder = VanillaEncoder(
                node_features, hidden_features, num_layers, rnn_hidden, latent_dim
            )

        # --- Decoder ---
        self.decoder = Decoder(
            latent_dim, seq_len, self.object_dim, self.joint_dim,
            self.agent, hidden_dim, obs_dim, fps
        )

        self.dt = 1.0 / float(fps)
        self.surrogate = SurrogateDynamics(
            joint_dim=self.joint_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            dt=self.dt,
        )


    # ---------- Latent sampling ----------
    @staticmethod
    def reparameterize_gaussian(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def reparameterize_gmm(mu, logvar, pi_logits, training: bool):
        # mu/logvar: [B, K, D], pi_logits: [B, K]
        pi = F.softmax(pi_logits, dim=-1)  # [B,K]
        # sample / argmax component
        if training:
            comp = torch.multinomial(pi, num_samples=1).squeeze(-1)  # [B]
        else:
            comp = torch.argmax(pi, dim=-1)  # [B]
        B = mu.shape[0]
        idx = torch.arange(B, device=mu.device)
        mu_sel = mu[idx, comp]        # [B,D]
        logv_sel = logvar[idx, comp]  # [B,D]
        std = torch.exp(0.5 * logv_sel)
        eps = torch.randn_like(std)
        return mu_sel + eps * std

    # ---------- Heteroscedastic NLL (per-step) ----------
    @staticmethod
    def hetero_nll_per_step(x, mu, log_sigma):
        """
        x, mu, log_sigma: [B,T,D] in same (normalized) space.
        Returns per-step NLL averaged over features: [B,T].
        """
        # per-element NLL
        inv_var = torch.exp(-2.0 * log_sigma)     # 1/σ^2
        elem = 0.5 * ((x - mu) ** 2) * inv_var + log_sigma + 0.5 * math.log(2 * math.pi)  # [B,T,D]
        return elem.mean(dim=-1)  # [B,T]


    def predict_dynamics(self, actions_seq, obs0, q0, dq0, mask):
        """
        actions_seq: [B, T, d]  (Trainer passes DETACHED actions so sim loss
                                 only updates this surrogate)
        obs0: [B, obs_dim], q0: [B, d], dq0: [B, d]
        mask: [B, T, 1] float {0,1}
        Returns:
          sur_obs_seq: [B, T, obs_dim]
          sur_q_seq:   [B, T, d]
          sur_dq_seq:  [B, T, d]
        """
        B, T, d = actions_seq.shape
        dev = actions_seq.device

        # Buffers/limits from decoder (already registered there)
        q_lower = self.decoder.joint_lower
        q_upper = self.decoder.joint_upper
        q_lower = q_lower.to(dev)
        q_upper = q_upper.to(dev)

        obs_seq_out = []
        q_seq_out = []
        dq_seq_out = []

        # Current state
        q_t = q0.to(dev)
        dq_t = dq0.to(dev)
        obs_t = obs0.to(dev)

        for t in range(T):
            mask_t = mask[:, t, :]  # [B,1]
            a_t = actions_seq[:, t, :]

            # One-step surrogate
            q_pred, dq_pred, obs_pred = self.surrogate(q_t, dq_t, obs_t, a_t)

            # Clamp q to joint limits (post integration)
            q_pred = torch.clamp(q_pred, q_lower, q_upper)

            # Freeze beyond padding
            q_t_next  = mask_t * q_pred  + (1.0 - mask_t) * q_t
            dq_t_next = mask_t * dq_pred + (1.0 - mask_t) * dq_t
            obs_next  = mask_t * obs_pred + (1.0 - mask_t) * obs_t

            q_seq_out.append(q_t_next)
            dq_seq_out.append(dq_t_next)
            obs_seq_out.append(obs_next)

            q_t, dq_t, obs_t = q_t_next, dq_t_next, obs_next

        sur_q_seq   = torch.stack(q_seq_out, dim=1)    # [B,T,d]
        sur_dq_seq  = torch.stack(dq_seq_out, dim=1)   # [B,T,d]
        sur_obs_seq = torch.stack(obs_seq_out, dim=1)  # [B,T,obs_dim]
        return sur_obs_seq, sur_q_seq, sur_dq_seq
    # ---------- Forward ----------
    def forward(
            self,
            x,
            edge_index,
            mask,  # [B,T,1] float {0,1}
            normalizer,
            obs_seq=None,  # [B,T,obs_dim] or None
            q=None,  # [B,T,d] (teacher joints) or None
            dq=None,  # [B,T,d] (teacher vels) or None
            tf_ratio: float = 1.0,
    ):
        """
        Returns (for both Gaussian and GMM priors):
          recon_mu:   [B,T,(links+1)*3]  normalized FK (links) + object
          joint_cmd:  [B,T,d]            predicted joints
          actions:    [B,T,d]            decoder actions in [-1,1]
          log_sigma:  [B,T,(links+1)*3]  predicted log sigmas over pose dims
          ... + latent tuple per prior
        """
        B, T = mask.shape[0], mask.shape[1]
        dev = x.device
        d = self.joint_dim
        obj_dim = self.object_dim

        # ----- encode -> z -----
        if self.prior == "GMM":
            mu, logvar, pi_logits = self.encoder(x, edge_index, mask)  # mu/logvar: [B,K,D], pi: [B,K]
            z = self.reparameterize_gmm(mu, logvar, pi_logits, self.training)  # [B,D]
        else:
            mu, logvar = self.encoder(x, edge_index, mask)  # [B,D]
            z = self.reparameterize_gaussian(mu, logvar)  # [B,D]

        # ----- prepare sequences -----
        if obs_seq is None:
            obs_seq = torch.zeros(B, T, self.decoder.obs_dim, device=dev)

        actions_seq, joints_seq, objects_seq, logsig_seq = [], [], [], []

        # Running state: start at default q, zero dq, and obs from first frame (or zeros)
        q_prev  = self.decoder.default_dof_pos.unsqueeze(0).expand(B, -1)  # [B,d]
        dq_prev = torch.zeros(B, d, device=dev)
        obs_state = obs_seq[:, 0, :] if obs_seq is not None else torch.zeros(B, self.decoder.obs_dim, device=dev)
        obs_state = normalizer(obs_state)
        for t in range(T):
            mask_t = mask[:, t, :]  # [B,1]

            # ---- Teacher forcing on inputs (same logic as before) ----
            if self.training and (q is not None):
                tf_gate = (torch.rand(1, device=dev) < tf_ratio).float().view(1, 1)  # [1,1]
                q_in = tf_gate * (q[:, t - 1, :] if t > 0 else self.decoder.default_dof_pos.unsqueeze(0)) \
                       + (1.0 - tf_gate) * q_prev
                dq_in = tf_gate * (dq[:, t - 1, :] if (t > 0 and dq is not None) else 0.0) \
                        + (1.0 - tf_gate) * dq_prev
            else:
                q_in, dq_in = q_prev, dq_prev

            # ---- One-step policy (use rolled obs_state, not GT obs[t]) ----
            action_t, obj_t, log_sig_t, _ = self.decoder(
                z, q_in, dq_in, obs_t=obs_state, mask_t=mask_t
            )  # action_t:[B,d], obj_t:[B,obj_dim], log_sig_t:[B,pos_dim]

            # ---- Surrogate dynamics step (replaces alpha-integration) ----
            q_pred, dq_pred, obs_pred = self.surrogate(q_in, dq_in, obs_state, action_t)
            obs_pred = normalizer(obs_pred)
            # Clamp joints to limits
            q_pred = torch.clamp(q_pred, self.decoder.joint_lower, self.decoder.joint_upper)

            # Freeze beyond padding (no drift on padded frames)
            q_new  = mask_t * q_pred  + (1.0 - mask_t) * q_prev
            dq_new = mask_t * dq_pred + (1.0 - mask_t) * dq_prev
            obs_new = mask_t * obs_pred + (1.0 - mask_t) * obs_state

            # Store masked outputs
            actions_seq.append(mask_t * action_t)
            joints_seq.append(q_new)
            objects_seq.append(mask_t * obj_t)   # object head still supervised by VAE loss
            logsig_seq.append(log_sig_t)         # mask applied in loss

            # Advance state
            q_prev, dq_prev, obs_state = q_new, dq_new, obs_new

        actions_seq = torch.stack(actions_seq, dim=1)  # [B,T,d]
        joints_seq  = torch.stack(joints_seq,  dim=1)  # [B,T,d]
        objects_seq = torch.stack(objects_seq, dim=1)  # [B,T,obj_dim]
        logsig_seq  = torch.stack(logsig_seq,  dim=1)  # [B,T,(links+1)*3]


        # ----- FK + normalization → recon_mu -----
        B, T, d = joints_seq.shape
        joint_flat = joints_seq.reshape(B * T, d).float()
        pos_flat = self.decoder.fk_model(joint_flat)  # [B*T, links*3]
        agent_traj = pos_flat.view(B, T, -1).float()  # [B,T, links*3]
        combined = torch.cat([agent_traj, objects_seq], dim=-1)  # [B,T,(links+1)*3]
        comb_resh = combined.view(B, T, -1, 3)

        # Use the normalization buffers (dataset stats should be loaded there)
        recon_mu = ((comb_resh - self.pos_mean) / self.pos_std).view(B, T, -1)

        # ----- return tuple consistent with your trainer -----
        if self.prior == "GMM":
            return recon_mu, joints_seq, actions_seq, logsig_seq, mu, logvar, pi_logits
        else:
            return recon_mu, joints_seq, actions_seq, logsig_seq, z, mu, logvar

    # ---------- Loss (masked, correct normalization) ----------
    def loss(
        self,
        recon_mu,          # [B,T,D] normalized FK positions (prediction)
        log_sigma,         # [B,T,D_log] (same D if you predict per-dim; else broadcastable)
        orig_traj,         # [B,T,D] normalized FK positions (target)
        act_mu,            # [B,T,A] teacher actions (target)
        action_seq,        # [B,T,A] decoder actions (prediction)
        mask,              # [B,T,1] float {0,1}
        *args,             # latent tuples per prior
        beta: float,
        lambda_kinematic: float = 1.0,
        lambda_dynamic: float = 0.2,
    ):
        """
        Masked sequence loss:
          - Pose: MSE (default) or heteroscedastic NLL per-step, masked and normalized.
          - Action: MSE per-step, masked and normalized, scaled by lambda_action.
          - KL: per-sequence.
        """
        # Ensure shapes
        B, T, D = recon_mu.shape
        mask_bt = mask.squeeze(-1)                         # [B,T], float in {0,1}
        valid = mask_bt.sum().clamp_min(1.0)               # scalar count of valid steps (across batch)
        orig_traj=orig_traj.view(B,T,-1)
        # -------- Pose reconstruction --------
        #pose_step = self.hetero_nll_per_step(orig_traj, recon_mu, log_sigma)
        # Per-step MSE averaged over feature dim: [B,T]
        pose_step = ((recon_mu - orig_traj) ** 2).mean(dim=-1)

        pose_loss = (pose_step * mask_bt).sum() / valid
        pose_loss = lambda_kinematic * pose_loss

        # -------- Action reconstruction --------
        # Per-step MSE on actions [B,T]
        act_step = ((action_seq - act_mu) ** 2).mean(dim=-1)
        action_loss = (act_step * mask_bt).sum() / valid
        action_loss = lambda_dynamic * action_loss

        total_recon = pose_loss + action_loss

        # -------- KL (per prior) --------
        if self.prior == "GMM":
            mu, logvar, pi_logits = args
            # mu/logvar: [B,K,D], prior_mu/logvar: [K,D]
            Bk, K, Dk = mu.shape
            pi = F.softmax(pi_logits, dim=-1)  # [B,K]

            prior_mu = self.prior_mu.unsqueeze(0).expand(Bk, -1, -1)         # [B,K,D]
            prior_logv = self.prior_logvar.unsqueeze(0).expand(Bk, -1, -1)   # [B,K,D]

            # KL of Gaussians q(z|x,c) || p(z|c), diagonal
            # 0.5 * [ log|Σ_p| - log|Σ_q| - D + tr(Σ_p^{-1} Σ_q) + (μ_q-μ_p)^T Σ_p^{-1} (μ_q-μ_p) ]
            kl_gauss = 0.5 * torch.sum(
                prior_logv - logvar - 1.0
                + (logvar - prior_logv).exp()
                + (mu - prior_mu).pow(2) / prior_logv.exp(),
                dim=-1,  # over D
            )  # [B,K]

            # Categorical KL: q(c|x) || p(c) with uniform prior
            log_q = torch.log(pi + 1e-10)
            log_p = math.log(1.0 / K)
            kl_cat = torch.sum(pi * (log_q - log_p), dim=-1)  # [B]

            kl_loss = torch.sum(pi * kl_gauss, dim=-1) + kl_cat  # [B]
            kl_loss = kl_loss.mean()

        else:  # Gaussian
            z, mu, logvar = args
            kl_per = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B]
            kl_loss = kl_per.mean()

        vae_loss = total_recon + beta * kl_loss
        return vae_loss, pose_loss, action_loss, kl_loss
