# Model/MoE.py
import math
from typing import List

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class MoEActorCritic(ActorCritic):
    """
    Mixture-of-Experts ActorCritic that is API-compatible with rsl_rl==2.2.4.

    - Inherits ActorCritic to preserve PPO's expected interface.
    - Replaces the actor mean with a mixture over frozen expert ActorCritics.
    - Adds a command head (0..1) -> scaled to env ranges; first num_cmd obs dims
      are overwritten with normalized commands (shared, and masked per expert).
    - Single critic: same critic MLP as base class, but run on critic_obs where
      the first num_cmd dims are replaced by normalized (shared) command.

    Required inputs to __init__:
      experts_ac: list of frozen ActorCritic modules (expert.actor_critic)
      num_cmd:    number of command dims at the front of the observation
      cmd_lows/highs: tensors [C] with env command ranges
      cmd_masks:  tensor [K, C] (0/1), per-expert visibility of each command
      gate_hidden: list of hidden sizes for the gate MLP
      topk:       select top-k experts (default 2)
      stickiness: EMA coefficient for routing smoothness (0..1)
      obs_norm:   optional normalizer with .mean/.var/.epsilon; if None/Identity, passthrough
    """

    def __init__(
        self,
        *,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        experts_ac: List[ActorCritic],
        num_cmd: int,
        cmd_lows: torch.Tensor,
        cmd_highs: torch.Tensor,
        cmd_masks: torch.Tensor,             # [K, C] with 0/1
        gate_hidden: List[int] = (256, 256),
        activation: str = "elu",
        topk: int = 2,
        stickiness: float = 0.85,
        obs_norm=None,
        # keep base ActorCritic kwargs for noise/std configs etc.
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
    ):
        # Build the base nets (actor will be unused for mean; critic is used)
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        self.obs_dim = num_actor_obs
        self.act_dim = num_actions
        self.num_cmd = int(num_cmd)

        # Experts (ActorCritic modules), frozen
        self.experts: List[ActorCritic] = experts_ac
        self.K = len(self.experts)
        for ex in self.experts:
            ex.eval()
            for p in ex.parameters():
                p.requires_grad_(False)

        # Command range buffers
        assert cmd_lows.shape[0] == self.num_cmd and cmd_highs.shape[0] == self.num_cmd
        self.register_buffer("cmd_lows", cmd_lows.clone().detach())
        self.register_buffer("cmd_highs", cmd_highs.clone().detach())

        # Per-expert cmd masks [K, C]
        assert cmd_masks.shape == (self.K, self.num_cmd)
        self.register_buffer("cmd_masks", cmd_masks.clone().detach())

        # Normalizer (may be None or Identity)
        self.obs_norm = obs_norm

        # Gating & command heads
        act = resolve_nn_activation(activation)
        # command head -> [0,1]^C
        self.cmd_head = nn.Sequential(
            nn.Linear(num_actor_obs, gate_hidden[0]),
            act,
            nn.Linear(gate_hidden[0], self.num_cmd),
        )
        # gate over experts
        gate_layers = []
        prev = num_actor_obs
        for h in gate_hidden:
            gate_layers += [nn.Linear(prev, h), act]
            prev = h
        gate_layers += [nn.Linear(prev, self.K)]
        self.gate = nn.Sequential(*gate_layers)

        # routing configuration
        self.topk = int(topk)
        self.stickiness = float(stickiness)
        self.register_buffer("prev_w", None, persistent=False)  # [B,K] at runtime

        # rsl_rl PPO boilerplate flags (non-recurrent)
        self.is_recurrent = False
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # place holders used by PPO
        self.distribution: Normal | None = None
        self.last_cmd01 = None
        self.last_gate_w = None

    # ---------- helpers ----------
    def _scale_cmd01_to_range(self, cmd01: torch.Tensor) -> torch.Tensor:
        # cmd01 in [0,1] -> env units
        return self.cmd_lows + (self.cmd_highs - self.cmd_lows) * cmd01

    def _norm_cmd_full(self, cmd_full: torch.Tensor) -> torch.Tensor:
        on = self.obs_norm
        if on is None or not (hasattr(on, "mean") and hasattr(on, "var") and hasattr(on, "epsilon")):
            return cmd_full
        mean = on.mean[:self.num_cmd].to(cmd_full.device)
        var = on.var[:self.num_cmd].to(cmd_full.device)
        return (cmd_full - mean) / torch.sqrt(var + on.epsilon)

    def _apply_topk_sticky(self, w: torch.Tensor) -> torch.Tensor:
        # softmax already done
        if self.topk < self.K:
            topv, topi = torch.topk(w, self.topk, dim=-1)
            mask = torch.zeros_like(w).scatter(-1, topi, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-8)
        # sticky routing
        if self.stickiness > 0.0:
            if self.prev_w is None or self.prev_w.shape[0] != w.shape[0]:
                self.prev_w = torch.full_like(w, 1.0 / self.K)
            w = self.stickiness * self.prev_w + (1.0 - self.stickiness) * w
            self.prev_w = w.detach()
        return w

    def _mixture_mean(self, obs: torch.Tensor, sticky: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          mu_mix : [B, A]
          w      : [B, K]
          cmd01  : [B, C]
        """
        # 1) shared command
        cmd01 = torch.sigmoid(self.cmd_head(obs))             # [B,C] in [0,1]
        cmd_full = self._scale_cmd01_to_range(cmd01)          # [B,C] env units
        cmd_norm = self._norm_cmd_full(cmd_full)              # [B,C] normalized (if norm present)

        # 2) gating on obs with injected shared command
        gate_in = obs.clone()
        gate_in[:, :self.num_cmd] = cmd_norm
        w_logits = self.gate(gate_in)                         # [B,K]
        w = torch.softmax(w_logits, dim=-1)
        if sticky:
            w = self._apply_topk_sticky(w)
        else:
            # still apply topk at inference for crisp routing
            if self.topk < self.K:
                topv, topi = torch.topk(w, self.topk, dim=-1)
                mask = torch.zeros_like(w).scatter(-1, topi, 1.0)
                w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-8)

        # 3) query frozen experts (ActorCritic) to get μ_k
        mus = []
        for k, ex in enumerate(self.experts):
            cmd_k_full = cmd_full * self.cmd_masks[k].unsqueeze(0)  # [B,C]
            cmd_k_norm = self._norm_cmd_full(cmd_k_full)
            obs_k = obs.clone()
            obs_k[:, :self.num_cmd] = cmd_k_norm
            # prefer deterministic mean for a cleaner mixture
            if hasattr(ex, "act_inference"):
                mu_k = ex.act_inference(obs_k)
            else:
                # fallback: run distribution and read mean
                ex.update_distribution(obs_k)
                mu_k = ex.action_mean
            mus.append(mu_k)
        mus = torch.stack(mus, dim=1)                         # [B,K,A]
        mu_mix = (w.unsqueeze(-1) * mus).sum(dim=1)           # [B,A]
        return mu_mix, w, cmd01

    # ---------- API methods PPO expects ----------
    def update_distribution(self, observations: torch.Tensor):
        # Replace base mean with MoE mean
        mu_mix, w, cmd01 = self._mixture_mean(observations, sticky=True)
        # std per rsl_rl base (scalar or log)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mu_mix)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mu_mix)
        else:
            raise ValueError("Unknown std type")

        self.distribution = Normal(mu_mix, std)
        # book-keeping for env
        self.last_cmd01 = cmd01.detach()
        self.last_gate_w = w.detach()

    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor):
        mu_mix, _, _ = self._mixture_mean(observations, sticky=False)
        return mu_mix

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        # Inject the SAME shared command into the critic input
        cmd01 = torch.sigmoid(self.cmd_head(critic_observations))
        cmd_full = self._scale_cmd01_to_range(cmd01)
        cmd_norm = self._norm_cmd_full(cmd_full)

        v_in = critic_observations.clone()
        v_in[:, :self.num_cmd] = cmd_norm
        return self.critic(v_in)

    # Non-recurrent hooks PPO will touch
    def reset(self, dones=None):
        if self.prev_w is not None and dones is not None:
            if dones.ndim == 2:
                dones = dones.squeeze(-1)
            if dones.dtype != torch.bool:
                dones = dones > 0
            self.prev_w[dones] = 1.0 / self.K
