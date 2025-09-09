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
    Semi-MDP Top-K MoE ActorCritic (stateful spinal router) with per-expert sticky commands.

    First pass uses raw cmd01 (no 0.5 prior). If you pre-seed prev_cmd01 / prev_cmd01_k
    with the correct shapes, _ensure_router_state will NOT overwrite them.
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
        cmd_masks: torch.Tensor,
        gate_hidden: List[int] = (256, 256),
        activation: str = "elu",
        topk: int = 2,
        stickiness: float = 0.85,
        hysteresis: float = 0.5,
        dwell_min: int | List[int] = 6,
        cmd_stickiness: float = 0.9,
        obs_norm=None,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
    ):
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

        self.experts: List[ActorCritic] = experts_ac
        self.K = len(self.experts)
        for ex in self.experts:
            ex.eval()
            for p in ex.parameters():
                p.requires_grad_(False)

        assert cmd_lows.shape[0] == self.num_cmd and cmd_highs.shape[0] == self.num_cmd
        self.register_buffer("cmd_lows", cmd_lows.clone().detach())
        self.register_buffer("cmd_highs", cmd_highs.clone().detach())
        assert cmd_masks.shape == (self.K, self.num_cmd)
        self.register_buffer("cmd_masks", cmd_masks.clone().detach())

        self.obs_norm = obs_norm

        act = resolve_nn_activation(activation)
        self.cmd_head = nn.Sequential(
            nn.Linear(num_actor_obs, gate_hidden[0]),
            act,
            nn.Linear(gate_hidden[0], self.num_cmd),
        )
        gate_layers = []
        prev = num_actor_obs
        for h in gate_hidden:
            gate_layers += [nn.Linear(prev, h), act]
            prev = h
        gate_layers += [nn.Linear(prev, self.K)]
        self.gate = nn.Sequential(*gate_layers)

        self.topk = int(topk)
        self.stickiness = float(stickiness)
        self.hysteresis = float(hysteresis)
        self.cmd_stickiness = float(cmd_stickiness)

        # Stateful buffers (lazy init)
        self.register_buffer("prev_w", None, persistent=False)           # [B,K]
        self.register_buffer("active_mask", None, persistent=False)      # [B,K] bool
        self.register_buffer("dwell", None, persistent=False)            # [B,K] long
        self.register_buffer("prev_cmd01", None, persistent=False)       # [B,C] or None
        self.register_buffer("prev_cmd01_k", None, persistent=False)     # [B,K,C] or None

        if isinstance(dwell_min, int):
            dwell_vec = torch.full((self.K,), int(dwell_min), dtype=torch.long)
        else:
            dwell_vec = torch.tensor(dwell_min, dtype=torch.long)
            assert dwell_vec.numel() == self.K
        self.register_buffer("dwell_min", dwell_vec)

        self.is_recurrent = False
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        self.distribution: Normal | None = None
        self.last_cmd01 = None
        self.last_cmd01_k = None
        self.last_gate_w = None

    # ----- helpers -----
    def _scale_cmd01_to_range(self, cmd01: torch.Tensor) -> torch.Tensor:
        return self.cmd_lows + (self.cmd_highs - self.cmd_lows) * cmd01

    def _norm_cmd_full(self, cmd_full: torch.Tensor) -> torch.Tensor:
        on = self.obs_norm
        if on is None or not (hasattr(on, "mean") and hasattr(on, "var") and hasattr(on, "epsilon")):
            return cmd_full
        mean = on.mean[:self.num_cmd].to(cmd_full.device)
        var = on.var[:self.num_cmd].to(cmd_full.device)
        return (cmd_full - mean) / torch.sqrt(var + on.epsilon)

    def _ensure_router_state(self, B: int, device: torch.device, C: int):
        need_init = (self.prev_w is None) or (self.prev_w.shape[0] != B)
        if need_init:
            self.prev_w = torch.full((B, self.K), 1.0 / self.K, device=device)
            self.active_mask = torch.zeros((B, self.K), device=device, dtype=torch.bool)
            self.dwell = torch.zeros((B, self.K), device=device, dtype=torch.long)
            # DO NOT overwrite pre-seeded sticky commands if shapes already match
            if (self.prev_cmd01 is not None) and (self.prev_cmd01.shape != (B, C)):
                self.prev_cmd01 = None
            if (self.prev_cmd01_k is not None) and (self.prev_cmd01_k.shape != (B, self.K, C)):
                self.prev_cmd01_k = None

    def _sticky_topk_route(self, logits: torch.Tensor) -> torch.Tensor:
        B, K = logits.shape
        active = self.active_mask
        dwell = self.dwell

        locked = active & (dwell < self.dwell_min.view(1, K))
        maxz, _ = logits.max(dim=-1, keepdim=True)
        keep = active & (~locked) & (logits >= (maxz - self.hysteresis))

        base = locked | keep
        slots_left = self.topk - base.sum(dim=-1, keepdim=True)
        slots_left = torch.clamp(slots_left, min=0)

        vals, idx = torch.topk(logits, k=K, dim=-1)
        selected = base.clone()
        for r in range(K):
            cand = idx[:, r]
            not_sel = ~selected.gather(1, cand.view(B, 1)).squeeze(1)
            can_take = not_sel & (slots_left.squeeze(1) > 0)
            if can_take.any():
                selected.scatter_(1, cand.view(B, 1), can_take.view(B, 1))
                slots_left = slots_left - can_take.view(B, 1).to(slots_left.dtype)

        mask = selected.float()
        masked_logits = logits.masked_fill(~selected, -1e9)
        w_step = torch.softmax(masked_logits, dim=-1) * mask
        w_step = w_step / w_step.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.prev_w is None:
            w = w_step
        else:
            w = self.stickiness * self.prev_w + (1.0 - self.stickiness) * w_step
            w = (w * mask) / (w * mask).sum(dim=-1, keepdim=True).clamp_min(1e-8)

        self.active_mask = selected
        self.dwell = torch.where(selected, dwell + 1, torch.zeros_like(dwell))
        self.prev_w = w.detach()
        return w

    def _sticky_shared_cmd01(self, cmd01_now: torch.Tensor) -> torch.Tensor:
        if self.prev_cmd01 is None:
            self.prev_cmd01 = cmd01_now.detach()
            return cmd01_now
        cmd01 = self.cmd_stickiness * self.prev_cmd01 + (1.0 - self.cmd_stickiness) * cmd01_now
        self.prev_cmd01 = cmd01.detach()
        return cmd01

    def _sticky_per_expert_cmd01(self, cmd01_now: torch.Tensor) -> torch.Tensor:
        target = cmd01_now.unsqueeze(1) * self.cmd_masks.unsqueeze(0)  # [B,K,C]
        if self.prev_cmd01_k is None:
            self.prev_cmd01_k = target.detach()
            return target
        cmd01_k = self.cmd_stickiness * self.prev_cmd01_k + (1.0 - self.cmd_stickiness) * target
        self.prev_cmd01_k = cmd01_k.detach()
        return cmd01_k

    def _mixture_mean(self, obs: torch.Tensor):
        B = obs.shape[0]
        device = obs.device
        C = self.num_cmd
        self._ensure_router_state(B, device, C)

        cmd01_raw = torch.sigmoid(self.cmd_head(obs))               # [B,C] 0..1
        cmd01_sh = self._sticky_shared_cmd01(cmd01_raw)             # [B,C]

        cmd_full_sh = self._scale_cmd01_to_range(cmd01_sh)
        cmd_norm_sh = self._norm_cmd_full(cmd_full_sh)
        gate_in = obs.clone()
        gate_in[:, :self.num_cmd] = cmd_norm_sh
        w_logits = self.gate(gate_in)                               # [B,K]

        w = self._sticky_topk_route(w_logits)                       # [B,K]

        cmd01_k = self._sticky_per_expert_cmd01(cmd01_raw)          # [B,K,C]

        mus = []
        for k, ex in enumerate(self.experts):
            cmd_full_k = self._scale_cmd01_to_range(cmd01_k[:, k, :])
            cmd_norm_k = self._norm_cmd_full(cmd_full_k)
            obs_k = obs.clone()
            obs_k[:, :self.num_cmd] = cmd_norm_k
            if hasattr(ex, "act_inference"):
                mu_k = ex.act_inference(obs_k)
            else:
                ex.update_distribution(obs_k)
                mu_k = ex.action_mean
            mus.append(mu_k)
        mus = torch.stack(mus, dim=1)                                # [B,K,A]
        mu_mix = (w.unsqueeze(-1) * mus).sum(dim=1)                  # [B,A]
        return mu_mix, w, cmd01_sh, cmd01_k

    def update_distribution(self, observations: torch.Tensor):
        mu_mix, w, cmd01_sh, cmd01_k = self._mixture_mean(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mu_mix)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mu_mix)
        else:
            raise ValueError("Unknown std type")

        self.distribution = Normal(mu_mix, std)
        self.last_cmd01 = cmd01_sh.detach()
        self.last_cmd01_k = cmd01_k.detach()
        self.last_gate_w = w.detach()

    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor):
        mu_mix, _, _, _ = self._mixture_mean(observations)
        return mu_mix

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        B = critic_observations.shape[0]
        device = critic_observations.device
        self._ensure_router_state(B, device, self.num_cmd)

        cmd01_raw = torch.sigmoid(self.cmd_head(critic_observations))
        cmd01_sh = self._sticky_shared_cmd01(cmd01_raw)
        cmd_full = self._scale_cmd01_to_range(cmd01_sh)
        cmd_norm = self._norm_cmd_full(cmd_full)

        v_in = critic_observations.clone()
        v_in[:, :self.num_cmd] = cmd_norm
        return self.critic(v_in)

    def reset(self, dones=None):
        if dones is None:
            self.prev_w = None
            self.active_mask = None
            self.dwell = None
            # keep prev_cmd01 / prev_cmd01_k as-is (caller may reseed)
            return

        if dones.ndim == 2:
            dones = dones.squeeze(-1)
        if dones.dtype != torch.bool:
            dones = dones > 0

        if self.prev_w is not None:
            self.prev_w[dones] = 1.0 / self.K
        if self.active_mask is not None:
            self.active_mask[dones] = False
        if self.dwell is not None:
            self.dwell[dones] = 0
        # keep sticky commands; caller can explicitly reseed done envs if desired
