import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class Decoder(nn.Module):
    """
    One-step conditional policy with a PPO-friendly *unsquashed* action distribution.
    - Distribution: Normal(mean, std) over actions (no tanh).
    - forward() returns (action_t, object_t, log_sigma_pos_t, feats)
      where action_t = mean (deterministic head).
    - Helpers:
        * action_distribution(...)
        * sample_action(...)
        * log_prob(a, mean, log_std)
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
        std_init: float = 0.3,         # PPO-style state-independent log-std init
        std_min: float = 1e-4,
        std_max: float = 5.0,
        state_dependent_std: bool = True,  # flip to True if you later want a std head
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
        policy_input_dim = self.obs_dim + self.latent_dim
        self.feature_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # --- PPO-friendly (unsquashed) distribution head ---
        # mean (unbounded)
        self.action_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),
        )

        # std (log-std)
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.action_logstd_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.joint_dim),
            )
        else:
            log_std_init = math.log(std_init)
            self.action_logstd = nn.Parameter(torch.full((self.joint_dim,), log_std_init))

        # Object head (unchanged)
        self.object_head = nn.Linear(hidden_dim, self.object_dim)

        # Per-step variance over FK position vector (links+object)*3
        self.pos_dim = (len(self.fk_model.link_names) + 1) * 3
        self.var_head = nn.Linear(hidden_dim, self.pos_dim)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -2.0)  # start reasonably confident

        self.sigma_min = 0.05
        self.sigma_max = 0.5

        # std clamps for numerical stability if you use state-dependent std
        self._std_min = std_min
        self._std_max = std_max
        self._log_std_min = math.log(std_min)
        self._log_std_max = math.log(std_max)

        # caches (optional)
        self._last_mean = None
        self._last_log_std = None

    @property
    def alpha(self):
        """Per-joint integration gain in (0,1)."""
        return torch.sigmoid(self.alpha_logits)  # [d]

    # --------- core helpers for PPO wiring ---------
    def _compute_features(
        self,
        z: torch.Tensor,
        obs_t: Optional[torch.Tensor],
        mask_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        if obs_t is None:
            obs_t = torch.zeros(B, self.obs_dim, device=device)

        feats = self.feature_net(torch.cat([obs_t, z], dim=-1))
        if mask_t is not None:
            feats = feats * mask_t  # preserve your padded-step gating
        return feats

    def _dist_params_from_feats(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.action_mu_head(feats)  # unbounded
        if self.state_dependent_std:
            log_std = self.action_logstd_head(feats)
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        else:
            log_std = self.action_logstd.expand_as(mean)
        # cache (optional)
        self._last_mean = mean
        self._last_log_std = log_std
        return mean, log_std

    @staticmethod
    def _log_prob(a: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Log-prob under diagonal Normal N(mean, std) with no squashing.
        """
        var = torch.exp(2.0 * log_std)
        return (-0.5 * (((a - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)

    def action_distribution(
        self,
        z: torch.Tensor,
        obs_t: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean, log_std) for the (unsquashed) Normal.
        """
        feats = self._compute_features(z, obs_t, mask_t)
        mean, log_std = self._dist_params_from_feats(feats)
        return mean, log_std

    def sample_action(
        self,
        z: torch.Tensor,
        obs_t: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (a, u, log_prob) with a == u (no squashing).
        - If deterministic, u = mean.
        """
        mean, log_std = self.action_distribution(z, obs_t, mask_t)
        if deterministic:
            u = mean
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            u = mean + std * eps
        a = u
        logp = self._log_prob(a, mean, log_std)
        return a, u, logp

    def log_prob(self, a: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Public helper: log-prob under the (unsquashed) Normal.
        """
        return self._log_prob(a, mean, log_std)


    # --------- forward (deterministic head) ---------
    def forward(
        self,
        z: torch.Tensor,           # [B, Z]
        obs_t: torch.Tensor,       # [B, obs_dim]
        mask_t: Optional[torch.Tensor] = None  # [B, 1] float {0,1} for stats/regularization
    ):
        """
        Stateless one-step policy. Returns step-level predictions:
            action_t (deterministic) = mean (no squashing),
            object_t,
            log_sigma_pos_t,
            feats
        """
        feats = self._compute_features(z, obs_t, mask_t)
        mean, log_std = self._dist_params_from_feats(feats)
        action_t = mean
        object_t = self.object_head(feats)             # [B, obj]
        sigma_raw = self.var_head(feats)               # [B, pos_dim]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        log_sigma_pos_t = torch.log(sigma)             # [B, pos_dim]

        self._last_mean = mean
        self._last_log_std = log_std
        return action_t, object_t, log_sigma_pos_t, feats
