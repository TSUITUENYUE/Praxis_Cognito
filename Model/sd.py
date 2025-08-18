import math
import torch
import torch.nn as nn
from typing import Tuple

class SurrogateDynamics(nn.Module):
    """
    Pure state-transition surrogate:
        (q_t, dq_t, a_t, obj_t)  ->  (q_{t+1}, dq_{t+1}, obj_{t+1})

    - Agent state: q_t, dq_t   (joint positions/velocities)   [B, d]
    - Action:      a_t         (unsquashed; env should handle clipping/scales) [B, d]
    - Object:      obj_t       (concatenated [pos, vel])       [B, o], with o even
                               e.g., for a ball: pos(3), vel(3) -> o=6

    Design:
      * One shared trunk over [q, dq, a, obj] to learn cross-coupling.
      * Agent head predicts Δdq (bounded residual with learned per-joint scale).
      * Object heads predict Δv (velocity residual) and a tiny residual on position
        after Euler integration (helps small modeling gaps).

    NOTE: Joint limit clamping can be applied outside this module.
    """
    def __init__(self, joint_dim: int, obj_dim: int, hidden_dim: int, dt: float):
        super().__init__()
        assert obj_dim % 2 == 0, "obj_dim must be even: concat of [pos, vel]"
        self.d = joint_dim
        self.o = obj_dim
        self.dt = float(dt)

        in_dim = (3 * joint_dim) + obj_dim  # [q, dq, a] + [obj_pos, obj_vel]
        h = hidden_dim

        # Shared interaction trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        # ---------- Agent head: Δdq ----------
        self.dq_head = nn.Linear(h, joint_dim)
        # Per-joint learned scale (small at init)
        self.log_dq_scale = nn.Parameter(torch.full((joint_dim,), math.log(0.1)))
        nn.init.zeros_(self.dq_head.weight)
        nn.init.zeros_(self.dq_head.bias)

        # ---------- Object heads: Δv and residual Δp ----------
        self.dv_head   = nn.Linear(h, self.o)  # residual on velocity
        self.dp_res_head = nn.Linear(h, self.o)  # small residual on position after Euler
        self.log_dv_scale   = nn.Parameter(torch.full((self.o,), math.log(0.1)))
        self.log_dp_res_scale = nn.Parameter(torch.full((self.o,), math.log(0.01)))

        nn.init.zeros_(self.dv_head.weight)
        nn.init.zeros_(self.dv_head.bias)
        nn.init.zeros_(self.dp_res_head.weight)
        nn.init.zeros_(self.dp_res_head.bias)

    def forward(
        self,
        q_t: torch.Tensor,     # [B, d]
        dq_t: torch.Tensor,    # [B, d]
        u_t: torch.Tensor,     # [B, 3]
        v_t: torch.Tensor,     # [B, 3]
        a_t: torch.Tensor,     # [B, d]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_next   : [B, d]
            dq_next  : [B, d]
            obj_next : [B, o]  (concat of [pos_next, vel_next])
        """
        B, d = q_t.shape
        assert d == self.d, "joint_dim mismatch"


        # Shared features
        x = torch.cat([q_t, dq_t, u_t, v_t, a_t], dim=-1)
        h = self.trunk(x)

        # ----- Agent update -----
        # Δdq bounded via tanh, scaled per-joint (stable small residual dynamics)
        dq_delta = torch.tanh(self.dq_head(h)) * torch.exp(self.log_dq_scale)   # [B, d]
        dq_next  = dq_t + dq_delta
        q_next   = q_t + self.dt * dq_next  # clamp to joint limits outside if desired

        # ----- Object update -----
        # Velocity residual (learned)
        dv = torch.tanh(self.dv_head(h)) * torch.exp(self.log_dv_scale)          # [B, o/2]
        v_next = v_t + dv

        # Euler step for position + tiny residual to absorb modeling errors
        dp_res = torch.tanh(self.dp_res_head(h)) * torch.exp(self.log_dp_res_scale)  # [B, o/2]
        u_next = u_t + self.dt * v_next + dp_res

        return q_next, dq_next, u_next, v_next
