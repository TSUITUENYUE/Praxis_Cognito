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
        B, T, d = actions_seq.shape
        dev = actions_seq.device

        # joint limits on device (no-op if already there)
        q_lower = self.decoder.joint_lower.to(dev)
        q_upper = self.decoder.joint_upper.to(dev)

        # Preallocate outputs
        sur_obs_seq = torch.empty(B, T, self.surrogate.obs_dim, device=dev, dtype=obs0.dtype)
        sur_q_seq = torch.empty(B, T, d, device=dev, dtype=q0.dtype)
        sur_dq_seq = torch.empty(B, T, d, device=dev, dtype=dq0.dtype)

        # Current state (already on GPU)
        q_t = q0.to(dev, non_blocking=True)
        dq_t = dq0.to(dev, non_blocking=True)
        obs_t = obs0.to(dev, non_blocking=True)

        for t in range(T):
            mask_t = mask[:, t, :]  # [B,1]
            a_t = actions_seq[:, t, :]

            q_pred, dq_pred, obs_pred = self.surrogate(q_t, dq_t, obs_t, a_t)
            torch.clamp_(q_pred, q_lower, q_upper)  # in-place to save a buffer

            # Freeze beyond padding (no drift on padded frames)
            q_t = mask_t * q_pred + (1.0 - mask_t) * q_t
            dq_t = mask_t * dq_pred + (1.0 - mask_t) * dq_t
            obs_t = mask_t * obs_pred + (1.0 - mask_t) * obs_t

            # write outputs
            sur_q_seq[:, t].copy_(q_t)
            sur_dq_seq[:, t].copy_(dq_t)
            sur_obs_seq[:, t].copy_(obs_t)

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

        if obs_seq is None:
            obs_seq = torch.zeros(B, T, self.decoder.obs_dim, device=dev)

        # Pre-normalize teacher observations ONCE (used for the action distribution head)
        obs_tf_n_all = normalizer(obs_seq)  # [B,T,obs_dim]

        # Preallocate output tensors (match your original dtypes/shapes)
        pre_mu_seq = torch.empty(B, T, self.joint_dim, device=dev, dtype=obs_seq.dtype)
        log_std_seq = torch.empty(B, T, self.joint_dim, device=dev, dtype=obs_seq.dtype)
        actions_seq = torch.empty(B, T, self.joint_dim, device=dev, dtype=obs_seq.dtype)
        joints_seq = torch.empty(B, T, self.joint_dim, device=dev, dtype=obs_seq.dtype)
        objects_seq = torch.empty(B, T, self.object_dim, device=dev, dtype=obs_seq.dtype)
        logsig_seq = torch.empty(B, T, (self.agent.fk_model.num_nodes + 1) * 3, device=dev, dtype=obs_seq.dtype)

        # Running state: start at default q, zero dq, and obs from first frame (or zeros)
        q_prev = self.decoder.default_dof_pos.unsqueeze(0).expand(B, -1)  # [B,d]
        dq_prev = torch.zeros(B, self.joint_dim, device=dev, dtype=obs_seq.dtype)
        obs_state = obs_seq[:, 0, :]
        obs_state = normalizer(obs_state)

        # Teacher-forcing previous states
        if q is not None:
            q0 = self.decoder.default_dof_pos.unsqueeze(0).expand(B, -1)
            dq0 = torch.zeros(B, self.joint_dim, device=dev, dtype=obs_seq.dtype)
            q_tf_prev = torch.cat([q0.unsqueeze(1), q[:, :-1, :]], dim=1)  # [B,T,d]
            dq_tf_prev = torch.cat([dq0.unsqueeze(1), dq[:, :-1, :]], dim=1)  # [B,T,d]
        else:
            # NOTE: your original else-branch referenced joints_seq before it's defined.
            # To keep behavior unchanged and avoid a runtime error, we keep the same structure
            # but initialize a zero baseline here (only used when q is None).
            q_tf_prev = torch.zeros(B, T, self.joint_dim, device=dev, dtype=obs_seq.dtype)
            dq_tf_prev = torch.zeros_like(q_tf_prev)

        for t in range(T):
            mask_t = mask[:, t, :]  # [B,1]
            obs_tf_n = obs_tf_n_all[:, t, :]  # [B,obs_dim]

            # Teacher forcing gate
            if self.training and (q is not None):
                tf_gate = (torch.rand(1, device=dev) < tf_ratio).float().view(1, 1)
                q_in = tf_gate * (q[:, t - 1, :] if t > 0 else self.decoder.default_dof_pos.unsqueeze(0)) \
                       + (1.0 - tf_gate) * q_prev
                dq_in = tf_gate * (dq[:, t - 1, :] if (t > 0 and dq is not None) else 0.0) \
                        + (1.0 - tf_gate) * dq_prev
            else:
                q_in, dq_in = q_prev, dq_prev

            # Policy step
            action_t, obj_t, log_sig_t, _ = self.decoder(
                z, q_in, dq_in, obs_t=obs_state, mask_t=mask_t
            )

            # Surrogate dynamics step
            q_pred, dq_pred, obs_pred = self.surrogate(q_in, dq_in, obs_state, action_t)
            obs_pred = normalizer(obs_pred)
            torch.clamp_(q_pred, self.decoder.joint_lower, self.decoder.joint_upper)

            # Freeze beyond padding
            q_new = mask_t * q_pred + (1.0 - mask_t) * q_prev
            dq_new = mask_t * dq_pred + (1.0 - mask_t) * dq_prev
            obs_new = mask_t * obs_pred + (1.0 - mask_t) * obs_state

            # Action distribution (teacher-conditional)
            mean_t, log_std_t = self.decoder.action_distribution(
                z, q_tf_prev[:, t, :], dq_tf_prev[:, t, :], obs_tf_n, mask_t=mask_t
            )

            # Store step outputs
            pre_mu_seq[:, t].copy_(mean_t)
            log_std_seq[:, t].copy_(log_std_t)
            actions_seq[:, t].copy_(mask_t * action_t)
            joints_seq[:, t].copy_(q_new)
            objects_seq[:, t].copy_(mask_t * obj_t)
            logsig_seq[:, t].copy_(log_sig_t)

            # Advance
            q_prev, dq_prev, obs_state = q_new, dq_new, obs_new


        # ----- FK + normalization → recon_mu -----
        B, T, d = joints_seq.shape
        joint_flat = joints_seq.reshape(B * T, d)  # keep dtype from seq (AMP-friendly)
        pos_flat = self.decoder.fk_model(joint_flat)  # [B*T, links*3]
        agent_traj = pos_flat.view(B, T, -1)  # [B,T, links*3]
        combined = torch.cat([agent_traj, objects_seq], dim=-1)  # [B,T,(links+1)*3]
        comb_resh = combined.view(B, T, -1, 3)

        # Use buffers pos_mean/pos_std (broadcasted)
        recon_mu = ((comb_resh - self.pos_mean) / self.pos_std).reshape(B, T, -1)

        # ----- return tuple consistent with your trainer -----
        if self.prior == "GMM":
            return recon_mu, joints_seq, actions_seq, pre_mu_seq, log_std_seq, logsig_seq, mu, logvar, pi_logits
        else:
            return recon_mu, joints_seq, actions_seq, pre_mu_seq, log_std_seq, logsig_seq, z, mu, logvar

    # ---------- Loss (masked, correct normalization) ----------
    def loss(
        self,
        recon_mu,          # [B,T,D] normalized FK positions (prediction)
        log_sigma,         # [B,T,D_log] (same D if you predict per-dim; else broadcastable)
        orig_traj,         # [B,T,D] normalized FK positions (target)
        act_mu,            # [B,T,A] teacher actions (target)
        action_seq,        # [B,T,A] decoder actions (prediction)
        pre_mu_seq,           # [B,T,d]  (decoder mean, pre-tanh)
        log_std_seq,          # [B,T,d]  (decoder log-std)
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
        # pose loss
        pose_step = ((recon_mu - orig_traj) ** 2).mean(dim=-1)  # [B,T]
        pose_loss = (pose_step.mul(mask_bt)).sum().div(valid)  # fuse multiplications, no new tensors
        pose_loss = lambda_kinematic * pose_loss

        # action NLL (factor out constant once)
        LOG_2PI = math.log(2.0 * math.pi)

        def _normal_nll(a, mean, log_std):
            inv_var = torch.exp(-2.0 * log_std)  # 1/sigma^2
            # 0.5 * [ (a-mean)^2 / var + 2 log_std + log(2π) ]
            return 0.5 * ((a - mean) ** 2 * inv_var + 2.0 * log_std + LOG_2PI)

        nll_step = _normal_nll(act_mu, pre_mu_seq, log_std_seq).sum(dim=-1)  # [B,T]
        action_loss = (nll_step.mul(mask_bt)).sum().div(valid)
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
