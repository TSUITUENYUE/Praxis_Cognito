import math
import torch
import torch.nn as nn
from typing import Tuple

class SurrogateDynamics(nn.Module):
    """
    Command-free surrogate dynamics (single-step residual model).

    Inputs (shapes per batch B):
      q_t      : [B, d]    joint positions
      dq_t     : [B, d]    joint velocities (q̇)
      a_t      : [B, d]    actions (unsquashed; env handles limits/scales)
      p_t      : [B, 3]    base position
      dp_t  : [B, 3]    base linear velocity (ṗ)
      w_t      : [B, 3]    base angular velocity (ω)
      u_t      : [B, o]    object position
      du_t  : [B, o]    object velocity

    Outputs:
      q_next, dq_next, p_next, dp_next, w_next, u_next, du_next
    """
    def __init__(self, joint_dim: int, obj_dim: int, hidden_dim: int, dt: float):
        super().__init__()
        self.d  = int(joint_dim)
        self.o  = int(obj_dim)   # per-component dim for both u and du
        self.dt = float(dt)

        # feature dim: [q, dq, a] + [p, dp, w] + [u, du]
        in_dim = (3 * self.d) + 9 + (2 * self.o)
        h = hidden_dim

        # -------- shared trunk --------
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        # -------- joints: residual on dq (q̇) --------
        self.dq_head = nn.Linear(h, self.d)
        self.log_dq_scale = nn.Parameter(torch.full((self.d,), math.log(0.1)))
        nn.init.zeros_(self.dq_head.weight); nn.init.zeros_(self.dq_head.bias)

        # -------- object: residuals on u̇ and tiny position residual --------
        self.du_res_head = nn.Linear(h, self.o)  # Δu̇
        self.u_res_head     = nn.Linear(h, self.o)  # tiny p-residual after Euler
        self.log_du_res_scale = nn.Parameter(torch.full((self.o,), math.log(0.1 * self.dt)))
        self.log_u_res_scale     = nn.Parameter(torch.full((self.o,), math.log(0.01 * self.dt**2)))
        nn.init.zeros_(self.du_res_head.weight); nn.init.zeros_(self.du_res_head.bias)
        nn.init.zeros_(self.u_res_head.weight);     nn.init.zeros_(self.u_res_head.bias)

        # -------- base: residuals on ṗ and ω, tiny position residual --------
        self.dp_res_head = nn.Linear(h, 3)  # Δṗ
        self.w_res_head     = nn.Linear(h, 3)  # Δω
        self.p_res_head     = nn.Linear(h, 3)  # tiny p-residual after Euler
        self.log_dp_res_scale = nn.Parameter(torch.full((3,), math.log(0.05 * self.dt)))
        self.log_w_res_scale     = nn.Parameter(torch.full((3,), math.log(0.05 * self.dt)))
        self.log_p_res_scale     = nn.Parameter(torch.full((3,), math.log(0.005 * self.dt**2)))
        nn.init.zeros_(self.dp_res_head.weight); nn.init.zeros_(self.dp_res_head.bias)
        nn.init.zeros_(self.w_res_head.weight);     nn.init.zeros_(self.w_res_head.bias)
        nn.init.zeros_(self.p_res_head.weight);     nn.init.zeros_(self.p_res_head.bias)

    def forward(
        self,
        q_t:      torch.Tensor,  # [B, d]
        dq_t:     torch.Tensor,  # [B, d]
        p_t:      torch.Tensor,  # [B, 3]
        dp_t:  torch.Tensor,  # [B, 3]
        w_t:      torch.Tensor,  # [B, 3]
        u_t:      torch.Tensor,  # [B, o]
        du_t:  torch.Tensor,  # [B, o]
        a_t:      torch.Tensor,  # [B, d]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, d = q_t.shape
        assert d == self.d, "joint_dim mismatch"
        assert u_t.shape[-1] == self.o and du_t.shape[-1] == self.o, "obj_dim mismatch"

        # shared features
        x = torch.cat([q_t, dq_t, a_t, p_t, dp_t, w_t, u_t, du_t], dim=-1)
        h = self.trunk(x)

        # ----- joints -----
        dq_delta  = torch.tanh(self.dq_head(h)) * torch.exp(self.log_dq_scale)     # [B, d]
        dq_next   = dq_t + dq_delta
        q_next    = q_t  + self.dt * dq_next   # clamp to joint limits outside if desired

        # ----- object -----
        du_res = torch.tanh(self.du_res_head(h)) * torch.exp(self.log_du_res_scale)  # [B, o]
        du_next= du_t + du_res
        u_res     = torch.tanh(self.u_res_head(h)) * torch.exp(self.log_u_res_scale)          # [B, o]
        u_next    = u_t + self.dt * du_next + u_res

        # ----- base -----
        dp_res = torch.tanh(self.dp_res_head(h)) * torch.exp(self.log_dp_res_scale)  # [B, 3]
        dp_next= dp_t + dp_res
        w_res     = torch.tanh(self.w_res_head(h)) * torch.exp(self.log_w_res_scale)          # [B, 3]
        w_next    = w_t + w_res
        p_res     = torch.tanh(self.p_res_head(h)) * torch.exp(self.log_p_res_scale)          # [B, 3]
        p_next    = p_t + self.dt * dp_next + p_res

        return q_next, dq_next, p_next, dp_next, w_next, u_next, du_next
