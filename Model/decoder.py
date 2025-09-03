import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# Model/decoder_moe.py
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

class DecoderMoE(nn.Module):
    """
    VAE decoder with Mixture-of-Experts policy head (unsquashed Normal).
    - Experts: frozen ActorCritic primitives (same as data-gen).
    - Gate + command head read [obs, z].
    - Per-expert command slice injected via cmd_masks.
    - Distribution: Normal(mean=mixture_of_experts, std from head or param).
    - API mirrors Decoder and adds act(z, obs, mask, deterministic=False).
    """

    def __init__(
        self,
        latent_dim: int,
        seq_len: int,                 # kept for symmetry; unused
        object_dim: int,
        joint_dim: int,
        agent,
        hidden_dim: int,
        obs_dim: int = 72,
        fps: int = 30,
        std_init: float = 0.3,
        std_min: float = 1e-4,
        std_max: float = 5.0,
        state_dependent_std: bool = True,
        # MoE bits (same as data-gen)
        experts_ac=None,             # list[ActorCritic] (frozen primitives)
        num_cmd: int = 16,
        cmd_lows: torch.Tensor = None,   # [C]
        cmd_highs: torch.Tensor = None,  # [C]
        cmd_masks: torch.Tensor = None,  # [K, C]
        gate_hidden=(256, 256),
        topk: int = 2,
        obs_norm=None,               # normalizer used to normalize first C cmd dims
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jl = torch.as_tensor(self.agent.joint_limits_lower, device=device, dtype=torch.float32)
        ju = torch.as_tensor(self.agent.joint_limits_upper, device=device, dtype=torch.float32)
        self.register_buffer("joint_lower", jl)
        self.register_buffer("joint_upper", ju)
        self.register_buffer("joint_range", (ju - jl) / 2.0)
        self.register_buffer("joint_mean",  (ju + jl) / 2.0)
        init_angles = torch.as_tensor(self.agent.init_angles, device=device, dtype=torch.float32)
        self.register_buffer("default_dof_pos", init_angles)

        self.alpha_logits = nn.Parameter(torch.full((joint_dim,), math.log(0.3)))
        self.action_scale = getattr(self.agent, "action_scale", 0.25)

        # feature net only used for variance head (+ optional state-dependent std)
        policy_input_dim = self.obs_dim + self.latent_dim
        self.feature_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),       nn.LayerNorm(hidden_dim), nn.ReLU(),
        )

        # --- MoE parts (identical structure to data-gen) ---
        self.experts = nn.ModuleList(experts_ac)
        for ex in self.experts:
            ex.eval()
            for p in ex.parameters():
                p.requires_grad_(False)

        self.num_cmd = num_cmd
        self.register_buffer("cmd_lows",  cmd_lows.clone().detach())
        self.register_buffer("cmd_highs", cmd_highs.clone().detach())
        self.register_buffer("cmd_masks", cmd_masks.clone().detach())
        self.obs_norm = obs_norm
        self.K = len(self.experts)
        self.topk = int(topk)

        gate_in = self.obs_dim + self.latent_dim
        self.cmd_head = nn.Sequential(
            nn.Linear(gate_in, gate_hidden[0]), nn.ELU(),
            nn.Linear(gate_hidden[0], self.num_cmd),
        )
        gh = []
        prev = gate_in
        for h in gate_hidden:
            gh += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        gh += [nn.Linear(prev, self.K)]
        self.gate = nn.Sequential(*gh)

        # --- distribution heads (unsquashed Normal) ---
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.action_logstd_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, self.joint_dim),
            )
        else:
            log_std_init = math.log(std_init)
            self.action_logstd = nn.Parameter(torch.full((self.joint_dim,), log_std_init))

        # per-step variance for FK position vector (links+object)*3
        self.pos_dim = (len(self.fk_model.link_names) + 1) * 3
        self.var_head = nn.Linear(hidden_dim, self.pos_dim)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -2.0)

        self._log_std_min = math.log(std_min)
        self._log_std_max = math.log(std_max)

        # caches
        self._last_mean = None
        self._last_log_std = None
        self.last_cmd01 = None
        self.last_gate_w = None

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logits)

    # ---------- MoE helpers (same math as data-gen) ----------
    def _scale01(self, x: torch.Tensor):
        return self.cmd_lows + (self.cmd_highs - self.cmd_lows) * x

    def _norm_cmd(self, cmd_full: torch.Tensor) -> torch.Tensor:
        on = self.obs_norm
        if on is None or not (hasattr(on, "mean") and hasattr(on, "var") and hasattr(on, "epsilon")):
            return cmd_full
        mean = on.mean[:self.num_cmd].to(cmd_full.device)
        var = on.var[:self.num_cmd].to(cmd_full.device)
        return (cmd_full - mean) / torch.sqrt(var + on.epsilon)

    def _topk(self, w: torch.Tensor):
        if self.topk >= self.K:
            return w
        topv, topi = torch.topk(w, self.topk, dim=-1)
        mask = torch.zeros_like(w).scatter(-1, topi, 1.0)
        return (w * mask) / (w * mask).sum(-1, keepdim=True)

    def _mixture_mean(self, obs: torch.Tensor, z: torch.Tensor):
        oz = torch.cat([obs, z], dim=-1)                     # [B, obs+z]
        cmd01 = torch.sigmoid(self.cmd_head(oz))             # [B,C] in [0,1]
        cmd_full = self._scale01(cmd01)
        cmd_norm = self._norm_cmd(cmd_full)

        gate_obs = obs.clone()
        gate_obs[:, :self.num_cmd] = cmd_norm
        w = torch.softmax(self.gate(torch.cat([gate_obs, z], dim=-1)), dim=-1)
        w = self._topk(w)

        mus = []
        for k, ex in enumerate(self.experts):
            obs_k = obs.clone()
            obs_k[:, :self.num_cmd] = cmd_norm * self.cmd_masks[k].unsqueeze(0)
            mu_k = ex.act_inference(obs_k)                   # [B,A] mean of expert
            mus.append(mu_k)
        mus = torch.stack(mus, dim=1)                        # [B,K,A]
        mu_mix = (w.unsqueeze(-1) * mus).sum(dim=1)          # [B,A]
        self.last_cmd01 = cmd01.detach()
        self.last_gate_w = w.detach()
        return mu_mix

    def get_cmd(self):
        return self._scale01(self.last_cmd01)
    # ---------- distribution helpers ----------
    def _compute_feats(self, z: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor):
        feats = self.feature_net(torch.cat([obs, z], dim=-1))
        return feats * mask

    def _dist_params(self, mu_mix: torch.Tensor, feats: torch.Tensor):
        if self.state_dependent_std:
            log_std = self.action_logstd_head(feats)
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        else:
            log_std = self.action_logstd.expand_as(mu_mix)
        self._last_mean = mu_mix
        self._last_log_std = log_std
        return mu_mix, log_std

    @staticmethod
    def _log_prob(a: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor):
        var = torch.exp(2.0 * log_std)
        return (-0.5 * (((a - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)

    # ---------- public API ----------
    def action_distribution(self, z: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor):
        mu_mix = self._mixture_mean(obs, z)
        feats  = self._compute_feats(z, obs, mask)
        mean, log_std = self._dist_params(mu_mix, feats)
        return mean, log_std

    def act(self, z: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.action_distribution(z, obs, mask)
        if deterministic:
            a = mean
        else:
            std = torch.exp(log_std)
            a = mean + std * torch.randn_like(std)
        logp = self._log_prob(a, mean, log_std)
        return a, mean, log_std, logp

    def forward(self, z: torch.Tensor, obs_t: torch.Tensor, mask_t: torch.Tensor):
        mean, log_std = self.action_distribution(z, obs_t, mask_t)
        action_t = mean
        feats = self._compute_feats(z, obs_t, mask_t)
        sigma_raw = self.var_head(feats)
        sigma = 0.05 + (0.5 - 0.05) * torch.sigmoid(sigma_raw)
        log_sigma_pos_t = torch.log(sigma)
        return action_t, mean, log_std, log_sigma_pos_t, feats



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
        # self.object_head = nn.Linear(hidden_dim, self.object_dim)

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
        feats = self.feature_net(torch.cat([obs_t, z], dim=-1))
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
        mask_t: torch.Tensor       # [B, 1] float {0,1} for stats/regularization
    ):
        """
        Stateless one-step policy. Returns step-level predictions:
            action_t (deterministic) = mean (no squashing),
            log_sigma_pos_t,
            feats
        """
        feats = self._compute_features(z, obs_t, mask_t)
        mean, log_std = self._dist_params_from_feats(feats)
        action_t = mean
        # object_t = self.object_head(feats)             # [B, obj]
        sigma_raw = self.var_head(feats)               # [B, pos_dim]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        log_sigma_pos_t = torch.log(sigma)             # [B, pos_dim]

        self._last_mean = mean
        self._last_log_std = log_std
        return action_t, mean, log_std, log_sigma_pos_t, feats
