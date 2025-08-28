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

    Outputs:
      q_next, dq_next, p_next, dp_next, w_next, dw_next, u_next, du_next
    """
    def __init__(
        self,
        joint_dim: int,
        obj_dim: int,
        hidden_dim: int,
        dt: float,
        agent,                    # to access fk (base frame) + per-link radii
        ball_radius: float = 0.05,
        sdf_tau: float = 0.02     # softness for SDF gating (meters)
    ):
        super().__init__()
        self.d  = int(joint_dim)
        self.o  = int(obj_dim)  # typically 3 for position/velocity
        self.dt = float(dt)

        # FK returns link positions in BASE frame; we rotate+translate to WORLD inside forward().
        self.fk = agent.fk_model
        link_r = getattr(agent, "link_bsphere_radius", None)
        self.register_buffer("link_bsphere_radius", link_r.clone())

        self.ball_radius = float(ball_radius)
        self.sdf_tau     = float(sdf_tau)

        # Feature dim = [q,dq,a] + [p(3),dp(3),quat(4),dw(3)] + [u(o),du(o)]
        in_dim = (3 * self.d) + 13 + (2 * self.o)
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

        # -------- base: residuals on ṗ and ω (WORLD), tiny position residual --------
        self.dp_res_head = nn.Linear(h, 3)  # Δṗ (WORLD)
        self.dw_res_head = nn.Linear(h, 3)  # Δω  (WORLD)
        self.p_res_head  = nn.Linear(h, 3)  # tiny position residual (WORLD)
        self.log_dp_res_scale = nn.Parameter(torch.full((3,), math.log(0.05 * self.dt)))
        self.log_dw_res_scale = nn.Parameter(torch.full((3,), math.log(0.05 * self.dt)))
        self.log_p_res_scale  = nn.Parameter(torch.full((3,), math.log(0.005 * self.dt**2)))
        nn.init.zeros_(self.dp_res_head.weight); nn.init.zeros_(self.dp_res_head.bias)
        nn.init.zeros_(self.dw_res_head.weight); nn.init.zeros_(self.dw_res_head.bias)
        nn.init.zeros_(self.p_res_head.weight);  nn.init.zeros_(self.p_res_head.bias)

        # -------- object: residuals on u̇ (WORLD) + tiny pos residual --------
        self.du_res_head = nn.Linear(h, self.o)  # Δu̇
        self.u_res_head  = nn.Linear(h, self.o)  # tiny position residual
        self.log_du_res_scale = nn.Parameter(torch.full((self.o,), math.log(0.1 * self.dt)))
        self.log_u_res_scale  = nn.Parameter(torch.full((self.o,), math.log(0.01 * self.dt**2)))
        nn.init.zeros_(self.du_res_head.weight); nn.init.zeros_(self.du_res_head.bias)
        nn.init.zeros_(self.u_res_head.weight);  nn.init.zeros_(self.u_res_head.bias)

        # -------- contact gating + impulse heads --------
        self.contact_gate = nn.Sequential(
            nn.Linear(h + 2, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 2)  # logits for [ball↔links, base↔ground]
        )
        self.impulse_ball_head    = nn.Linear(h, 3)      # Δv to ball from link–ball
        self.impulse_ground_head  = nn.Linear(h, 3)      # Δv to base from base–ground
        self.impulse_bg_head      = nn.Linear(h, self.o) # Δv to ball from ball–ground (o=3)

        self.log_imp_ball_scale    = nn.Parameter(torch.full((3,),      math.log(0.2)))
        self.log_imp_ground_scale  = nn.Parameter(torch.full((3,),      math.log(0.2)))
        self.log_imp_bg_scale      = nn.Parameter(torch.full((self.o,), math.log(0.2)))

        nn.init.zeros_(self.impulse_ball_head.weight);   nn.init.zeros_(self.impulse_ball_head.bias)
        nn.init.zeros_(self.impulse_ground_head.weight); nn.init.zeros_(self.impulse_ground_head.bias)
        nn.init.zeros_(self.impulse_bg_head.weight);     nn.init.zeros_(self.impulse_bg_head.bias)

    # ---------------- quaternion utils ----------------
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
        """WORLD ω → small rotation quaternion for dt."""
        theta = (omega * self.dt).norm(dim=-1, keepdim=True)
        half  = 0.5 * theta
        k = torch.where(theta > 1e-8, torch.sin(half) / theta, 0.5 - (theta**2)/48.0)
        vec = k * (omega * self.dt)
        dq  = torch.cat([torch.cos(half), vec], dim=-1)
        return self._quat_normalize(dq)

    @staticmethod
    def _rotate_vec_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rotate v [B,L,3] by quaternion q [B,4] (w,x,y,z)."""
        B, L, _ = v.shape
        w,x,y,z = q.unbind(-1)
        qw = w.view(B,1,1); qx = x.view(B,1,1); qy = y.view(B,1,1); qz = z.view(B,1,1)
        tx = 2*(qy*v[...,2:3] - qz*v[...,1:2])
        ty = 2*(qz*v[...,0:1] - qx*v[...,2:3])
        tz = 2*(qx*v[...,1:2] - qy*v[...,0:1])
        vpx = v[...,0:1] + qw*tx + (qy*tz - qz*ty)
        vpy = v[...,1:2] + qw*ty + (qz*tx - qx*tz)
        vpz = v[...,2:3] + qw*tz + (qx*ty - qy*tx)
        return torch.cat([vpx, vpy, vpz], dim=-1)

    # ---------------- SDF helpers ----------------
    def _sdf_ball(self, link_pos_w: torch.Tensor, ball_center_w: torch.Tensor) -> torch.Tensor:
        """Min signed distance (link-sphere ↔ ball-sphere) over links."""
        B, L, _ = link_pos_w.shape
        c_ball = ball_center_w.view(B,1,3).expand(B,L,3)
        d = torch.linalg.norm(link_pos_w - c_ball, dim=-1) - (self.link_bsphere_radius.view(1,L) + self.ball_radius)
        return d.min(dim=1).values  # [B]

    def _sdf_ground(self, link_pos_w: torch.Tensor) -> torch.Tensor:
        """Ground plane z=0; link-sphere SDF = z - r. Returns min over links."""
        z = link_pos_w[..., 2]  # [B,L]
        d = z - self.link_bsphere_radius.view(1, -1)
        return d.min(dim=1).values

    # ---------------- forward ----------------
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
        a_t:   torch.Tensor,  # [B, d]  torque
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, d = q_t.shape
        assert d == self.d, "joint_dim mismatch"
        assert w_t.shape[-1] == 4, "w_t must be quaternion [w,x,y,z]"
        assert u_t.shape[-1] == self.o and du_t.shape[-1] == self.o, "obj_dim mismatch"

        # shared features (WORLD-frame states)
        x = torch.cat([q_t, dq_t, a_t, p_t, dp_t, w_t, dw_t, u_t, du_t], dim=-1)
        h = self.trunk(x)

        # ----- joints -----
        dq_delta = torch.tanh(self.dq_head(h)) * torch.exp(self.log_dq_scale)
        dq_next  = dq_t + dq_delta
        q_next   = q_t  + self.dt * dq_next

        # ----- base linear (WORLD) -----
        dp_res  = torch.tanh(self.dp_res_head(h)) * torch.exp(self.log_dp_res_scale)
        dp_next = dp_t + dp_res
        p_res   = torch.tanh(self.p_res_head(h))  * torch.exp(self.log_p_res_scale)
        p_next  = p_t + self.dt * dp_next + p_res

        # ----- base angular (WORLD ω): q_next = dQ(ω_world) ⊗ q_t -----
        dw_res      = torch.tanh(self.dw_res_head(h)) * torch.exp(self.log_dw_res_scale)
        dw_next     = dw_t + dw_res
        dquat_world = self._delta_quat_from_omega(dw_next)
        w_next      = self._quat_normalize(self._quat_mul(dquat_world, w_t))

        # ----- object linear (WORLD) -----
        du_res  = torch.tanh(self.du_res_head(h)) * torch.exp(self.log_du_res_scale)
        du_next = du_t + du_res
        u_res   = torch.tanh(self.u_res_head(h))  * torch.exp(self.log_u_res_scale)
        u_next  = u_t + self.dt * du_next + u_res

        # ---------- contact-aware impulses ----------
        # ball–ground gate (plane z=0 via ball center SDF)
        sdf_bg  = u_t[..., 2] - self.ball_radius                          # [B]
        gate_bg = torch.sigmoid(-sdf_bg / self.sdf_tau).unsqueeze(-1)     # [B,1]
        dv_bg   = torch.tanh(self.impulse_bg_head(h)) * torch.exp(self.log_imp_bg_scale)  # [B,o]
        du_next = du_next + gate_bg * dv_bg

        # link positions: FK in BASE → rotate by w_t and translate by p_t to WORLD
        link_pos_base  = self.fk(q_t).view(B, -1, 3)                       # [B,L,3] base
        link_pos_world = self._rotate_vec_by_quat(link_pos_base, w_t) + p_t.view(B,1,3)

        # SDFs for link–ball and base–ground
        sdf_ball   = self._sdf_ball(link_pos_world, u_t)   # [B]
        sdf_ground = self._sdf_ground(link_pos_world)      # [B]

        # geometric gates
        ball_gate_geom = torch.sigmoid(-sdf_ball   / self.sdf_tau).unsqueeze(-1)  # [B,1]
        base_gate_geom = torch.sigmoid(-sdf_ground / self.sdf_tau).unsqueeze(-1)  # [B,1]

        # learned confidence from features + SDF scalars
        feat = torch.cat([h, sdf_ball.unsqueeze(-1), sdf_ground.unsqueeze(-1)], dim=-1)
        contact_logits = self.contact_gate(feat)                 # [B,2]
        contact_conf   = torch.sigmoid(contact_logits)           # [B,2]

        # final gates
        ball_gate = ball_gate_geom * contact_conf[:, 0:1]        # link–ball
        base_gate = base_gate_geom * contact_conf[:, 1:2]        # base–ground

        # impulses (consistent scaling; NO cap)
        imp_ball   = torch.tanh(self.impulse_ball_head(h))   * torch.exp(self.log_imp_ball_scale)     # [B,3]
        imp_ground = torch.tanh(self.impulse_ground_head(h)) * torch.exp(self.log_imp_ground_scale)   # [B,3]

        du_next = du_next + ball_gate * imp_ball   # kick to ball
        dp_next = dp_next + base_gate * imp_ground # kick to base

        return q_next, dq_next, p_next, dp_next, w_next, dw_next, u_next, du_next
