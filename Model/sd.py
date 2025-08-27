import math
import torch
import torch.nn as nn
from typing import Tuple

class SurrogateDynamics(nn.Module):
    """
    Command-free surrogate dynamics (single-step residual model, WORLD-frame states).

    Inputs (per batch B):
      q_t   : [B, d]   joint positions
      dq_t  : [B, d]   joint velocities
      a_t   : [B, d]   actions (unsquashed)
      p_t   : [B, 3]   base position (WORLD)
      dp_t  : [B, 3]   base linear velocity (WORLD)
      w_t   : [B, 4]   base orientation quaternion (w,x,y,z)
      dw_t  : [B, 3]   base angular velocity ω (WORLD)
      u_t   : [B, o]   object position (WORLD)      -- expect o=3
      du_t  : [B, o]   object linear vel (WORLD)    -- expect o=3
      dv_t  : [B, 3]   object angular vel (WORLD)   -- sphere spin; optional but supported

    Outputs:
      q_next, dq_next, p_next, dp_next, w_next, dw_next, u_next, du_next, dv_next
    """
    def __init__(self, joint_dim: int, obj_dim: int, hidden_dim: int, dt: float):
        super().__init__()
        self.d  = int(joint_dim)
        self.o  = int(obj_dim)  # typically 3 for position/velocity
        self.dt = float(dt)

        # feature dim = [q,dq,a] + [p(3),dp(3),quat(4),dw(3)] + [u(o),du(o),dv(3)]
        in_dim = (3 * self.d) + 13 + (2 * self.o )
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

        # -------- base: residuals on ṗ (WORLD) and ω (WORLD), tiny position residual --------
        self.dp_res_head = nn.Linear(h, 3)  # Δṗ (WORLD)
        self.dw_res_head = nn.Linear(h, 3)  # Δω  (WORLD)
        self.p_res_head  = nn.Linear(h, 3)  # tiny position residual (WORLD)
        self.log_dp_res_scale = nn.Parameter(torch.full((3,), math.log(0.05 * self.dt)))
        self.log_dw_res_scale = nn.Parameter(torch.full((3,), math.log(0.05 * self.dt)))
        self.log_p_res_scale  = nn.Parameter(torch.full((3,), math.log(0.005 * self.dt**2)))
        nn.init.zeros_(self.dp_res_head.weight); nn.init.zeros_(self.dp_res_head.bias)
        nn.init.zeros_(self.dw_res_head.weight); nn.init.zeros_(self.dw_res_head.bias)
        nn.init.zeros_(self.p_res_head.weight);  nn.init.zeros_(self.p_res_head.bias)

        # -------- object: residuals on u̇ (WORLD), tiny pos residual, and ω_obj (WORLD) --------
        self.du_res_head = nn.Linear(h, self.o)  # Δu̇
        self.u_res_head  = nn.Linear(h, self.o)  # tiny position residual
        self.dv_res_head = nn.Linear(h, 3)       # Δω_obj (WORLD)
        self.log_du_res_scale = nn.Parameter(torch.full((self.o,), math.log(0.1 * self.dt)))
        self.log_u_res_scale  = nn.Parameter(torch.full((self.o,), math.log(0.01 * self.dt**2)))
        self.log_dv_res_scale = nn.Parameter(torch.full((3,),     math.log(0.05 * self.dt)))
        nn.init.zeros_(self.du_res_head.weight); nn.init.zeros_(self.du_res_head.bias)
        nn.init.zeros_(self.u_res_head.weight);  nn.init.zeros_(self.u_res_head.bias)
        nn.init.zeros_(self.dv_res_head.weight); nn.init.zeros_(self.dv_res_head.bias)

    @staticmethod
    def _quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Hamilton product: q ⊗ r, both [...,4] with (w,x,y,z)."""
        w1,x1,y1,z1 = q.unbind(-1)
        w2,x2,y2,z2 = r.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    @staticmethod
    def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    def _delta_quat_from_omega(self, omega: torch.Tensor) -> torch.Tensor:
        """
        omega: [...,3] angular velocity in a chosen frame (here WORLD).
        Returns a small rotation quaternion dq (w,x,y,z) for dt=self.dt.
        """
        theta = (omega * self.dt).norm(dim=-1, keepdim=True)          # [...,1]
        half  = 0.5 * theta
        # stable sin(x)/x near 0
        k = torch.where(theta > 1e-8, torch.sin(half) / theta, 0.5 - (theta**2)/48.0)
        vec = k * (omega * self.dt)                                   # [...,3]
        dq  = torch.cat([torch.cos(half), vec], dim=-1)               # [...,4]
        return self._quat_normalize(dq)

    def forward(
        self,
        q_t:   torch.Tensor,  # [B, d]
        dq_t:  torch.Tensor,  # [B, d]
        p_t:   torch.Tensor,  # [B, 3]  WORLD
        dp_t:  torch.Tensor,  # [B, 3]  WORLD
        w_t:   torch.Tensor,  # [B, 4]  orientation
        dw_t:  torch.Tensor,  # [B, 3]  WORLD angular vel
        u_t:   torch.Tensor,  # [B, o]  WORLD (o typically 3)
        du_t:  torch.Tensor,  # [B, o]  WORLD
        a_t:   torch.Tensor,  # [B, d]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, d = q_t.shape
        assert d == self.d, "joint_dim mismatch"
        assert w_t.shape[-1] == 4, "w_t must be quaternion [w,x,y,z]"
        assert u_t.shape[-1] == self.o and du_t.shape[-1] == self.o, "obj_dim mismatch"


        # shared features (WORLD-frame states)
        x = torch.cat([q_t, dq_t, a_t, p_t, dp_t, w_t, dw_t, u_t, du_t], dim=-1)
        h = self.trunk(x)

        # ----- joints -----
        dq_delta = torch.tanh(self.dq_head(h)) * torch.exp(self.log_dq_scale)   # [B, d]
        dq_next  = dq_t + dq_delta
        q_next   = q_t  + self.dt * dq_next

        # ----- base linear (WORLD) -----
        dp_res  = torch.tanh(self.dp_res_head(h)) * torch.exp(self.log_dp_res_scale)  # [B, 3]
        dp_next = dp_t + dp_res
        p_res   = torch.tanh(self.p_res_head(h))  * torch.exp(self.log_p_res_scale)   # [B, 3]
        p_next  = p_t + self.dt * dp_next + p_res

        # ----- base angular (WORLD ω): q_next = dQ(ω_world) ⊗ q_t -----
        dw_res      = torch.tanh(self.dw_res_head(h)) * torch.exp(self.log_dw_res_scale)  # [B, 3]
        dw_next     = dw_t + dw_res
        dquat_world = self._delta_quat_from_omega(dw_next)                                 # [B,4]
        w_next      = self._quat_normalize(self._quat_mul(dquat_world, w_t))

        # ----- object linear (WORLD) -----
        du_res  = torch.tanh(self.du_res_head(h)) * torch.exp(self.log_du_res_scale)  # [B, o]
        du_next = du_t + du_res
        u_res   = torch.tanh(self.u_res_head(h))  * torch.exp(self.log_u_res_scale)   # [B, o]
        u_next  = u_t + self.dt * du_next + u_res

        # ----- object angular (WORLD) -----
        #dv_res  = torch.tanh(self.dv_res_head(h)) * torch.exp(self.log_dv_res_scale)  # [B, 3]
        #dv_next = dv_t + dv_res

        return q_next, dq_next, p_next, dp_next, w_next, dw_next, u_next, du_next