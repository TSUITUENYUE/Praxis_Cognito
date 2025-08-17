import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Surrogate dynamics: one-step model f(q_t, dq_t, obs_t, a_t) → dq_{t+1}, Δobs
# Euler integration: q_{t+1} = clamp(q_t + dt * dq_{t+1}, joint_limits)
# ---------------------------------------------------------------------
class SurrogateDynamics(nn.Module):
    def __init__(self, joint_dim: int, obs_dim: int, hidden_dim: int, dt: float):
        super().__init__()
        self.joint_dim = joint_dim
        self.obs_dim = obs_dim
        self.dt = dt

        in_dim = joint_dim + joint_dim + obs_dim + joint_dim  # q, dq, obs, action
        h = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        # Δdq head (tanh + learned per-joint scale)
        self.dq_head = nn.Linear(h, joint_dim)
        # Δobs head (unconstrained; we scale small at init)
        self.obs_head = nn.Linear(h, obs_dim)

        # Per-joint learned scale for Δdq, initialized small for stability
        self.log_dq_scale = nn.Parameter(torch.full((joint_dim,), math.log(0.1)))
        # Small init so first predictions are near-zero deltas
        nn.init.zeros_(self.dq_head.weight)
        nn.init.zeros_(self.dq_head.bias)
        nn.init.zeros_(self.obs_head.weight)
        nn.init.zeros_(self.obs_head.bias)

    def forward(self, q_t, dq_t, obs_t, a_t):
        """
        q_t, dq_t: [B, d]
        obs_t:    [B, obs_dim]
        a_t:      [B, d]
        Returns:
          q_next, dq_next, obs_next
        """
        x = torch.cat([q_t, dq_t, obs_t, a_t], dim=-1)
        h = self.net(x)

        # Δdq with bounded magnitude (tanh) and learned scale
        dq_delta = torch.tanh(self.dq_head(h)) * torch.exp(self.log_dq_scale)  # [B, d]
        dq_next = dq_t + dq_delta

        # Euler step for q
        q_next = q_t + self.dt * dq_next  # clamp happens outside with known limits

        # Δobs with small magnitude at init (weights are zero by init)
        obs_delta = self.obs_head(h)
        obs_next = obs_t + obs_delta

        q_next = q_next.clone()
        dq_next = dq_next.clone()
        obs_next = obs_next.clone()
        return q_next, dq_next, obs_next
