import torch
import torch.nn as nn
import math
from typing import Optional

class Decoder(nn.Module):
    """
    One-step conditional policy:
      a_t = π(q_{t-1}, dq_{t-1}, obs_t, z)
    Heads: action (tanh), object pred, variance (for pose NLL).
    No autoregression here; unrolling happens outside.
    """

    def __init__(
        self,
        latent_dim: int,
        seq_len: int,                  # kept for compatibility but unused inside
        object_dim: int,
        joint_dim: int,
        agent,
        hidden_dim: int,
        obs_dim: int = 51,
        fps: int = 30,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_dim  = joint_dim
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.obs_dim    = obs_dim
        self.agent      = agent
        self.fk_model   = self.agent.fk_model
        self.frame_rate = fps

        # Normalization buffers for FK output (used by rollout, not here)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jl = torch.as_tensor(self.agent.joint_limits_lower, device=device, dtype=torch.float32)
        ju = torch.as_tensor(self.agent.joint_limits_upper, device=device, dtype=torch.float32)
        self.register_buffer("joint_lower", jl)
        self.register_buffer("joint_upper", ju)
        self.register_buffer("joint_range", (ju - jl) / 2.0)
        self.register_buffer("joint_mean",  (ju + jl) / 2.0)

        init_angles = torch.as_tensor(self.agent.init_angles, device=device, dtype=torch.float32)
        self.register_buffer("default_dof_pos", init_angles)

        # α (per-joint integration gain) is learned here; rollout reads it
        self.alpha_logits = nn.Parameter(torch.full((joint_dim,), math.log(0.3)))
        self.action_scale = getattr(self.agent, "action_scale", 0.25)

        # One-step policy features
        policy_input_dim = (2 * self.joint_dim) + self.obs_dim + self.latent_dim
        self.feature_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),
            nn.Tanh(),
        )
        self.object_head = nn.Linear(hidden_dim, self.object_dim)

        # Per-step variance over FK position vector (links+object)*3
        self.pos_dim = (len(self.fk_model.link_names) + 1) * 3
        self.var_head = nn.Linear(hidden_dim, self.pos_dim)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -2.0)  # start reasonably confident

        self.sigma_min = 0.05
        self.sigma_max = 0.5

    @property
    def alpha(self):
        """Per-joint integration gain in (0,1)."""
        return torch.sigmoid(self.alpha_logits)  # [d]

    def forward(
        self,
        z: torch.Tensor,           # [B, Z]
        q_prev: torch.Tensor,      # [B, d]
        dq_prev: torch.Tensor,     # [B, d]
        obs_t: Optional[torch.Tensor] = None,  # [B, obs_dim]
        mask_t: Optional[torch.Tensor] = None  # [B, 1] float {0,1} for stats/regularization
    ):
        """
        Stateless one-step policy. Returns step-level predictions.
        Mask is optional here; autoregressive masking is handled in the rollout.
        If provided, we only use it to gate feature statistics (no learnable drift on padded steps).
        """
        B = z.size(0)
        device = z.device
        if obs_t is None:
            obs_t = torch.zeros(B, self.obs_dim, device=device)

        # Normalize joints for network input (NOT for FK)
        q_norm = (q_prev - self.joint_mean) / self.joint_range

        policy_input = torch.cat([q_norm, dq_prev, obs_t, z], dim=-1)
        feats = self.feature_net(policy_input)

        if mask_t is not None:
            # Light gating so LayerNorm/Dropout stats don't drift on padded steps
            feats = feats * mask_t

        action_t = self.action_head(feats)                          # [-1,1]
        object_t = self.object_head(feats)                          # [B, obj]
        sigma_raw = self.var_head(feats)                            # [B, pos_dim]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        log_sigma_pos_t = torch.log(sigma)                          # [B, pos_dim]

        return action_t, object_t, log_sigma_pos_t, feats
