import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import genesis
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat,  transform_quat_by_quat

from pathlib import Path
import copy
from rsl_rl.runners import OnPolicyRunner
from Pretrain.go2_env_icm import Go2Env   # same env used in data gen
import genesis as gs

from .encoder import GMMEncoder, VanillaEncoder
from .decoder import Decoder, DecoderMoE
from .sd import SurrogateDynamics
from Pretrain.utils import build_soft_masks_from_sigma
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch._dynamo as dynamo
from omegaconf import DictConfig, OmegaConf

@dynamo.disable
def _store_col(dst, t, src):
    # identical effect to: dst[:, t].copy_(src)
    dst[:, t].copy_(src)

@dynamo.disable
def _clamp_to_limits(x, lo, hi):
    # identical numerics to clamp_ but out-of-place & eager
    return torch.clamp(x, lo, hi)

class IntentionVAE(nn.Module):
    def __init__(
        self,
        prior: str,
        node_features: int,
        hidden_features: int,
        num_layers: int,
        rnn_hidden: int,
        latent_dim: int,
        seq_len: int,
        agent,
        hidden_dim: int,
        obs_dim: int,
        fps: int,
        cfg: DictConfig,
        num_components: int = 128,
    ):
        super().__init__()
        assert prior in {"Gaussian", "GMM"}, "Supported priors: 'Gaussian', 'GMM'"
        self.prior = prior
        self.num_components = num_components if prior == "GMM" else None
        self.latent_dim = latent_dim
        self.agent = agent
        self.object_dim = agent.object_dim
        self.joint_dim = agent.n_dofs
        self.urdf = agent.urdf

        self.cfg = cfg
        #self.env_cfg = cfg.env
        self.obs_scale = cfg.obs.obs_scales
        self.default_dof_pos = agent.init_angles
        self.simulate_action_latency = self.cfg.env.simulate_action_latency
        self.command_scale = 1.0
        self.base_init_pos = self.cfg.env.base_init_pos
        self.base_init_quat = self.cfg.env.base_init_quat
        self.kp = self.cfg.env.kp
        self.kd = self.cfg.env.kd
        self.last_actions = None
        # NOTE: replace these with dataset stats (pos_mean/std) before training.
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_std", torch.ones(3))

        # --- Encoder ---
        if self.prior == "GMM":
            self.encoder = GMMEncoder(
                node_features, hidden_features, num_layers, rnn_hidden, latent_dim, num_components
            )
            # GMM prior (uniform mixing, diagonal unit variance)
            prior_mu = torch.randn(num_components, latent_dim) * 5.0
            self.register_buffer("prior_mu", prior_mu)
            self.register_buffer("prior_logvar", torch.zeros(num_components, latent_dim))
            self.register_buffer("prior_pi", torch.full((num_components,), 1.0 / num_components))
        else:  # Gaussian
            self.encoder = VanillaEncoder(
                node_features, hidden_features, num_layers, rnn_hidden, latent_dim
            )

        # --- Decoder ---
        #self.decoder = Decoder(latent_dim, seq_len, self.object_dim, self.joint_dim,self.agent, hidden_dim, obs_dim, fps)

        train_cfg = OmegaConf.to_container(self.cfg.train, resolve=True)
        gs.init(logging_level="warning")
        # env (exactly like data-gen)
        env = Go2Env(
            num_envs=1,
            env_cfg=self.cfg.env,
            obs_cfg=self.cfg.obs,
            reward_cfg=self.cfg.reward,
            command_cfg=self.cfg.command,
            show_viewer=False,
            agent=self.agent,
        )
        # locate primitives root and collect checkpoints
        prims_root = self.cfg.train.log_dir
        prims_root = Path(prims_root)

        ckpts = [d / "model_400.pt" for d in sorted(prims_root.iterdir()) if (d / "model_400.pt").exists()]

        runner = OnPolicyRunner(env, train_cfg, str(prims_root), device=gs.device)

        # load PPO experts and freeze
        experts_ac = []
        for p in ckpts:
            runner.load(str(p))
            expert_alg = copy.deepcopy(runner.alg)
            for param in expert_alg.actor_critic.parameters():
                param.requires_grad = False
            expert_alg.actor_critic.eval()
            experts_ac.append(expert_alg.actor_critic)

        # command ranges (C = env.num_commands)
        C = env.num_commands
        cmd_lows = torch.tensor([
            self.cfg.command.lin_vel_x_range[0], self.cfg.command.lin_vel_y_range[0], self.cfg.command.ang_vel_range[0],
            self.cfg.command.dz_range[0], self.cfg.command.pitch_range[0], self.cfg.command.roll_range[0],
            self.cfg.command.ee_vx_range[0], self.cfg.command.ee_vy_range[0], self.cfg.command.ee_vz_range[0],
            self.cfg.command.imp_s_range[0],
            self.cfg.command.Fn_range[0], self.cfg.command.vt_range[0], self.cfg.command.phi_range[0],
            self.cfg.command.apex_h_range[0], self.cfg.command.psi_range[0], self.cfg.command.pitch_imp_range[0],
        ], dtype=torch.float32, device=self.pos_mean.device)
        cmd_highs = torch.tensor([
            self.cfg.command.lin_vel_x_range[1], self.cfg.command.lin_vel_y_range[1], self.cfg.command.ang_vel_range[1],
            self.cfg.command.dz_range[1], self.cfg.command.pitch_range[1], self.cfg.command.roll_range[1],
            self.cfg.command.ee_vx_range[1], self.cfg.command.ee_vy_range[1], self.cfg.command.ee_vz_range[1],
            self.cfg.command.imp_s_range[1],
            self.cfg.command.Fn_range[1], self.cfg.command.vt_range[1], self.cfg.command.phi_range[1],
            self.cfg.command.apex_h_range[1], self.cfg.command.psi_range[1], self.cfg.command.pitch_imp_range[1],
        ], dtype=torch.float32, device=self.pos_mean.device)

        # SAME masks as in data-gen (expects 5 primitives). If you trained a different K, adjust indices here.
        def _on(*idxs):
            m = torch.zeros(C, device=self.pos_mean.device)
            m[list(idxs)] = 1.0
            return m

        cmd_masks = torch.stack([
            _on(0, 1, 2),  # locomotion: vx, vy, wz
            _on(3, 4, 5),  # body pose: dz, pitch, roll
            _on(6, 7, 8, 9),  # swing: ee_vx, ee_vy, ee_vz, imp_s
            _on(10, 11, 12),  # contact hold: Fn, vt, phi
            _on(13, 14, 15),  # hop: apex_h, psi, pitch_imp
        ], dim=0)

        # finally, the VAE decoder becomes a MoE that takes (z, obs, mask) and mixes the SAME primitives
        self.decoder = DecoderMoE(
            latent_dim=latent_dim,
            seq_len=seq_len,
            object_dim=self.object_dim,
            joint_dim=self.joint_dim,
            agent=self.agent,
            hidden_dim=hidden_dim,
            obs_dim=obs_dim,
            fps=fps,
            std_init=0.3,
            std_min=1e-4,
            std_max=5.0,
            state_dependent_std=True,
            experts_ac=experts_ac,  # frozen primitives
            num_cmd=C,
            cmd_lows=cmd_lows,
            cmd_highs=cmd_highs,
            cmd_masks=cmd_masks,  # per-primitive command visibility
            gate_hidden=(256, 256),
            topk=2,
            obs_norm=runner.obs_normalizer,  # same normalizer used in data-gen
        ).to(self.pos_mean.device)

        self.dt = 1.0 / float(fps)
        self.control_dt = self.cfg.env.dt
        self.K = max(1, int(round(self.dt / self.control_dt)))
        self.surrogate = SurrogateDynamics(self.joint_dim,self.object_dim,hidden_dim,self.dt,self.agent)
        # cache end-effector indices (Python list) to avoid per-step host sync
        try:
            self.ee_idx = [int(i) for i in self.agent.end_effector.tolist()]
        except Exception:
            self.ee_idx = []

        # Hoisted constants/buffers to reduce per-step allocations
        dev = self.pos_mean.device
        self.register_buffer("g_world_down", torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=dev))
        self.register_buffer("z_world_up", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=dev))
        self.register_buffer("x_body_axis", torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=dev))

        # Per-DOF PD gains as buffers (shape [1,d])
        def _per_dof(val, d, device):
            t = torch.as_tensor(val, dtype=torch.float32, device=device)
            if t.ndim == 0:
                t = t.repeat(d)
            assert t.numel() == d, "kp/kd must be scalar or per-DOF list"
            return t.view(1, d)

        self.register_buffer("kp_dof", _per_dof(self.kp, self.joint_dim, dev))
        self.register_buffer("kd_dof", _per_dof(self.kd, self.joint_dim, dev))


    # ---------- Latent sampling ----------
    @staticmethod
    def reparameterize_gaussian(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def reparameterize_gmm(mu, logvar, pi_logits, training: bool):
        # mu/logvar: [B, K, D], pi_logits: [B, K]
        pi = F.softmax(pi_logits, dim=-1)  # [B,K]
        # sample / argmax component
        if training:
            comp = torch.multinomial(pi, num_samples=1).squeeze(-1)  # [B]
        else:
            comp = torch.argmax(pi, dim=-1)  # [B]
        B = mu.shape[0]
        idx = torch.arange(B, device=mu.device)
        mu_sel = mu[idx, comp]        # [B,D]
        logdu_sel = logvar[idx, comp]  # [B,D]
        std = torch.exp(0.5 * logdu_sel)
        eps = torch.randn_like(std)
        return mu_sel + eps * std

    # ---------- Heteroscedastic NLL (per-step) ----------
    @staticmethod
    def hetero_nll_per_step(x, mu, log_sigma):
        """
        x, mu, log_sigma: [B,T,D] in same (normalized) space.
        Returns per-step NLL averaged over features: [B,T].
        """
        # per-element NLL
        inv_var = torch.exp(-2.0 * log_sigma)     # 1/σ^2
        elem = 0.5 * ((x - mu) ** 2) * inv_var + log_sigma + 0.5 * math.log(2 * math.pi)  # [B,T,D]
        return elem.mean(dim=-1)  # [B,T]

    def _expand_to_dof(self, x, d, dev):
        """
        Expand a scalar or 1D array-like to shape [1, d] on device.
        If x is already a tensor of shape [*, d], broadcast-safe ops will work.
        """
        if isinstance(x, (int, float)):
            t = torch.tensor([x], device=dev, dtype=torch.float32)
            return t.view(1, 1).expand(1, d)
        t = torch.as_tensor(x, device=dev, dtype=torch.float32)
        if t.ndim == 1:
            return t.view(1, -1)
        return t  # assume broadcastable

    def _pd_torque(self, q, dq, q_des, dq_des=None):
        """
        Genesis-style PD: per-DOF
          tau = Kp*(q_des - q) + Kv*((dq_des or 0) - dq)
        Then clamp by optional force limits.
        Shapes:
          q, dq, q_des, dq_des: [B, d]
        Returns:
          tau_pd: [B, d]
        """
        dev = q.device
        B, d = q.shape

        # gains from cached per-DOF buffers
        kp = self.kp_dof.expand(B, d)
        kv = self.kd_dof.expand(B, d)

        if dq_des is None:
            dq_des = torch.zeros_like(dq)
        tau = kp * (q_des - q) + kv * (dq_des - dq)

        # optional clamp by force range if present (Genesis does this)
        fr = self.cfg.env.get("force_range", None)  # e.g. {"lower": -87.0, "upper": 87.0} or per-DOF list
        if fr is not None:
            f_lo = self._expand_to_dof(fr.get("lower", float("-inf")), d, dev).expand(B, d)
            f_hi = self._expand_to_dof(fr.get("upper", float("inf")), d, dev).expand(B, d)
            tau = torch.clamp(tau, f_lo, f_hi)

        return tau

    def predict_dynamics(self, a_t, q_t, dq_t, p_t, dp_t, w_t, dw_t, u_t, du_t, mask_t,cmd):
        B, d = a_t.shape
        dev = a_t.device

        # --- 1) latency & PD target (target stays constant during the K ticks) ---
        cmd_hold = torch.clamp(a_t, -self.cfg.env["clip_actions"], self.cfg.env["clip_actions"])
        exec_actions_last = cmd_hold  # for logging in obs

        q_k, dq_k = q_t, dq_t
        p_k, dp_k = p_t, dp_t
        w_k, dw_k = w_t, dw_t
        u_k, du_k = u_t, du_t

        if self.last_actions is None:
            self.last_actions = torch.zeros(B, self.joint_dim, device=dev)
        for k in range(self.K):
            if self.simulate_action_latency and k == 0:
                exec_actions = self.last_actions  # 1-tick delay on the first inner tick
            else:
                exec_actions = cmd_hold  # current command thereafter
            q_des = exec_actions * self.cfg.env["action_scale"] + self.default_dof_pos
            tau_pd = self._pd_torque(q_k, dq_k, q_des, dq_des=None)

            q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k= self.surrogate(q_k, dq_k, p_k, dp_k, w_k, dw_k, u_k, du_k, tau_pd)

            exec_actions_last = exec_actions

        # predicted (t + rollout_dt)
        q_pred, dq_pred = q_k, dq_k
        p_pred, dp_pred = p_k, dp_k
        w_pred, dw_pred = w_k, dw_k
        u_pred, du_pred = u_k, du_k

        # --- 3) mask EVERY state (freeze beyond padding) ---
        m = mask_t
        if m.dim() == 1:
            m = m.unsqueeze(-1)  # [B,1]
        q_next = m * q_pred + (1.0 - m) * q_t
        dq_next = m * dq_pred + (1.0 - m) * dq_t
        p_next = m * p_pred + (1.0 - m) * p_t
        dp_next = m * dp_pred + (1.0 - m) * dp_t
        w_next = m * w_pred + (1.0 - m) * w_t
        dw_next = m * dw_pred + (1.0 - m) * dw_t
        u_next = m * u_pred + (1.0 - m) * u_t
        du_next = m * du_pred + (1.0 - m) * du_t

        # --- 4) obs from the UPDATED state (body-frame vel assumed) ---
        inv_q_next = inv_quat(w_next)
        base_lin_vel = transform_by_quat(dp_next, inv_q_next)
        base_ang_vel = transform_by_quat(dw_next, inv_q_next)
        g_world = self.g_world_down.to(device=dev, dtype=q_t.dtype).expand(B, 3)
        proj_g = transform_by_quat(g_world, inv_q_next)
        relative_ball_pos = transform_by_quat(u_next - p_next, inv_q_next)
        relative_ball_vel = transform_by_quat(du_next - dp_next, inv_q_next)

        # --- Base height (world z) ---
        base_height = p_next[:, 2:3]

        # --- Contact flags via surrogate dynamics (per end-effector) ---
        # Approximate contact using signed distance of EE link-spheres to ground (z - radius).
        # Gate with a sigmoid and threshold to obtain binary flags, matching env format.
        with torch.no_grad():
            link_pos_base = self.surrogate.fk(q_next).view(B, -1, 3)  # [B,L,3] in BASE frame
            # rotate to WORLD and translate by base position
            L = link_pos_base.shape[1]
            link_flat = link_pos_base.reshape(B * L, 3)
            quat_rep = w_next.repeat_interleave(L, dim=0)
            link_world_flat = transform_by_quat(link_flat, quat_rep)
            link_pos_world = link_world_flat.view(B, L, 3) + p_next.view(B, 1, 3)

            ee_idx = self.ee_idx
            z_world_ee = link_pos_world[:, ee_idx, 2]  # [B, E]
            radii_ee = self.surrogate.link_bsphere_radius[ee_idx].view(1, -1).to(z_world_ee.device)
            sdf_ee = z_world_ee - radii_ee  # >0 above ground, <0 penetrating
            prob_ee = torch.sigmoid(-sdf_ee / max(1e-6, float(self.surrogate.sdf_tau)))
            contact_flags = (prob_ee > 0.5).to(z_world_ee.dtype)  # [B, E] as float

        obs_pred = torch.cat([
            cmd * self.command_scale,  # commands
            base_lin_vel * self.cfg.obs.obs_scales["lin_vel"],  # 3
            base_ang_vel * self.cfg.obs.obs_scales["ang_vel"],  # 3
            proj_g,  # 3
            base_height,  # 1 (inserted to match env format)
            (q_next - self.default_dof_pos) * self.cfg.obs.obs_scales["dof_pos"],  # d
            dq_next * self.cfg.obs.obs_scales["dof_vel"],  # d
            exec_actions_last,  # d
            contact_flags,  # E contact flags to mirror env (e.g., 4 feet)
            relative_ball_pos,  # 3
            relative_ball_vel  # 3
        ], dim=-1)

        # bookkeeping for next call
        self.last_actions = a_t.detach()
        self.last_dof_vel = dq_next.detach()

        # RETURN the masked "next" state (consistent with obs_pred)
        return obs_pred, q_next, dq_next, p_next, dp_next, w_next, dw_next, u_next, du_next

    def forward(
            self,
            x,
            edge_index,
            mask,  # [B,T,1] float {0,1}
            normalizer,
            obs_seq,  # [B,T,obs_dim] (INPUT)
            q,  # [B,T,d]
            dq,  # [B,T,d]
            p,  # [B,T,3]
            dp,  # [B,T,3]
            dw,  # [B,T,3]
            u,  # [B,T,3]
            du,  # [B,T,3]
            dv,  # [B,T,3]
            tf_ratio: float = 1.0,
    ):
        """
        Returns (for both Gaussian and GMM priors):
          recon_traj:   [B,T,(links+1)*3]  normalized FK (links) + object-pos(3)
          joint_cmd:  [B,T,d]            predicted joints
          actions:    [B,T,d]            decoder actions in [-1,1]
          log_sigma:  [B,T,(links+1)*3]  predicted log sigmas over pose dims
          ... + latent tuple per prior
        """
        B, T = mask.shape[0], mask.shape[1]
        dev = x.device

        # ---- keep the original observation tensor safe ----
        obs_in_seq = obs_seq  # [B,T,obs_dim]
        self.obs_dim = obs_in_seq.shape[-1]
        obs_dtype = obs_in_seq.dtype

        # ----- encode -> z -----
        if self.prior == "GMM":
            mu, logvar, pi_logits = self.encoder(x, edge_index, mask)  # mu/logvar: [B,K,D], pi: [B,K]
            z = self.reparameterize_gmm(mu, logvar, pi_logits, self.training)  # [B,D]
        else:
            mu, logvar, ptr = self.encoder(x, edge_index, mask)  # [B,D]
            z = self.reparameterize_gaussian(mu, logvar)  # [B,D]

        # Pre-normalize teacher observations ONCE (for decoder conditioning)
        obs_tf_n_all = normalizer(obs_in_seq)  # [B,T,obs_dim]

        # -------- Collect outputs per-timestep, stack once at the end --------
        actions_list, mu_list, log_std_list = [], [], []
        joints_list, dq_list = [], []
        w_list, dw_list = [], []
        p_list, dp_list = [], []
        obs_list, objects_list = [], []
        log_sig_list = []

        def _quat_from_g(proj_g: torch.Tensor) -> torch.Tensor:
            """
            proj_g: [B,3] gravity in the *body* frame (what your obs stores)
            returns: q_bw [B,4] body->world quaternion with "zero yaw" choice
            """
            eps = 1e-8
            B = proj_g.shape[0]
            dev = proj_g.device
            dtype = proj_g.dtype

            # World up/down from buffer
            z_w = self.z_world_up.to(device=dev, dtype=dtype).expand(B, 3)

            # Body 'up' unit vector in *body* frame (opposite gravity)
            a = -proj_g / (proj_g.norm(dim=-1, keepdim=True) + eps)  # [B,3]

            # Minimal rotation that takes a -> z_w (quaternion from two vectors)
            v = torch.cross(a, z_w, dim=-1)  # [B,3]
            c = (a * z_w).sum(dim=-1, keepdim=True)  # [B,1]
            s = torch.sqrt(torch.clamp(1.0 + c, min=0.0)) * 2.0  # [B,1]

            xyz = v / (s + eps)  # [B,3]
            w = 0.5 * s  # [B,1]
            q_align = torch.cat([w, xyz], dim=-1)  # [B,4]
            q_align = q_align / (q_align.norm(dim=-1, keepdim=True) + eps)

            # Resolve the free yaw: make body x-axis align with world x after rotation
            x_b = self.x_body_axis.to(device=dev, dtype=dtype).expand(B, 3)
            x0_w = transform_by_quat(x_b, q_align)  # [B,3]
            phi = torch.atan2(x0_w[:, 1], x0_w[:, 0])  # yaw of x0_w
            half = -0.5 * phi
            qz = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
                             dim=-1)  # [B,4]

            # Compose: q = qz ⊗ q_align  (Genesis' transform_quat_by_quat composes quats)
            q = transform_quat_by_quat(qz, q_align)  # [B,4]
            q = q / (q.norm(dim=-1, keepdim=True) + eps)
            return q

        # -------- Running state (init from GT t=0 so time aligns with obs_in_seq[:,0]) --------
        q_prev = q[:, 0, :].to(dev)  # [B,d]
        dq_prev = dq[:, 0, :].to(dev)  # [B,d]
        p_prev = p[:, 0, :].to(dev)  # [B,3]
        dp_prev = dp[:, 0, :].to(dev)  # [B,3]
        # reconstruct w_prev from projected gravity; gravity starts after [cmd(C), lin(3), ang(3)]
        cmd_dim = getattr(self.decoder, "num_cmd", None)
        if cmd_dim is None:
            cmd_dim = int(self.cfg.command.get("num_commands", 16))
        grav_start = cmd_dim + 6
        g0 = obs_in_seq[:, 0, grav_start:grav_start + 3].to(dev)  # [B,3] projected gravity at t=0
        w_prev = _quat_from_g(g0)  # [B,4]
        dw_prev = dw[:, 0, :].to(dev)  # [B,3]
        u_prev = u[:, 0, :].to(dev)  # [B,3]
        du_prev = du[:, 0, :].to(dev)  # [B,3]

        # Initial decoder observation (normalized)
        obs_prev = obs_tf_n_all[:, 0, :]  # [B,obs_dim]

        for t in range(T):
            mask_t = mask[:, t, :]  # [B,1]
            m = mask_t if mask_t.dim() == 2 else mask_t.unsqueeze(-1)

            # ----- Teacher Forcing (full state at time t) -----
            if self.training:
                tf_gate = (torch.rand(B, 1, device=dev) < tf_ratio).float()
                # GT for time t
                q_teacher = q[:, t, :]
                dq_teacher = dq[:, t, :]
                p_teacher = p[:, t, :]
                dp_teacher = dp[:, t, :]
                # reconstruct w_teacher from obs_in_seq gravity at time t
                g_t = obs_in_seq[:, t, grav_start:grav_start + 3].to(dev)
                w_teacher = _quat_from_g(g_t)
                dw_teacher = dw[:, t, :]
                u_teacher = u[:, t, :]
                du_teacher = du[:, t, :]

                # Blend GT vs model-prev; renormalize quaternion afterwards
                q_in = tf_gate * q_teacher + (1.0 - tf_gate) * q_prev
                dq_in = tf_gate * dq_teacher + (1.0 - tf_gate) * dq_prev
                p_in = tf_gate * p_teacher + (1.0 - tf_gate) * p_prev
                dp_in = tf_gate * dp_teacher + (1.0 - tf_gate) * dp_prev
                w_in = tf_gate * w_teacher + (1.0 - tf_gate) * w_prev
                w_in = w_in / (w_in.norm(dim=-1, keepdim=True) + 1e-8)
                dw_in = tf_gate * dw_teacher + (1.0 - tf_gate) * dw_prev
                u_in = tf_gate * u_teacher + (1.0 - tf_gate) * u_prev
                du_in = tf_gate * du_teacher + (1.0 - tf_gate) * du_prev
            else:
                q_in, dq_in = q_prev, dq_prev
                p_in, dp_in = p_prev, dp_prev
                # reconstruct current w_in from current obs_prev gravity (consistent)
                g_cur = obs_in_seq[:, t, grav_start:grav_start + 3].to(dev)
                w_in = _quat_from_g(g_cur)
                dw_in = dw_prev
                u_in, du_in = u_prev, du_prev

            # ----- Policy step (condition on last observation) -----
            action_t, mu_t, log_std_t, log_sig_t, _ = self.decoder(
                z, obs_t=obs_prev, mask_t=mask_t
            )
            cmd_t = self.decoder.get_cmd()
            # ----- Surrogate dynamics (FULL state → next state) -----
            obs_pred, q_pred, dq_pred, p_pred, dp_pred, w_pred, dw_pred, u_pred, du_pred = \
                self.predict_dynamics(action_t, q_in, dq_in, p_in, dp_in, w_in, dw_in, u_in, du_in, mask_t,cmd_t)

            # Clamp joints to limits
            q_pred = _clamp_to_limits(q_pred, self.decoder.joint_lower, self.decoder.joint_upper)

            # Freeze beyond padding (apply mask to ALL state parts)
            q_next = m * q_pred + (1.0 - m) * q_prev
            dq_next = m * dq_pred + (1.0 - m) * dq_prev
            p_next = m * p_pred + (1.0 - m) * p_prev
            dp_next = m * dp_pred + (1.0 - m) * dp_prev
            w_next = m * w_pred + (1.0 - m) * w_prev
            w_next = w_next / (w_next.norm(dim=-1, keepdim=True) + 1e-8)
            dw_next = m * dw_pred + (1.0 - m) * dw_prev
            u_next = m * u_pred + (1.0 - m) * u_prev
            du_next = m * du_pred + (1.0 - m) * du_prev

            # Observation for next policy step (already produced by predict_dynamics)
            obs_next = m * obs_pred + (1.0 - m) * obs_prev

            # ----- Store step outputs -----
            actions_list.append(m * action_t)
            mu_list.append(mu_t)
            log_std_list.append(log_std_t)
            joints_list.append(q_next)
            dq_list.append(dq_next)
            w_list.append(w_next)
            dw_list.append(dw_next)
            p_list.append(p_next)
            dp_list.append(dp_next)
            objects_list.append(u_next)
            obs_list.append(obs_next)
            log_sig_list.append(log_sig_t)

            # ----- Advance recurrent state -----
            q_prev, dq_prev = q_next, dq_next
            p_prev, dp_prev = p_next, dp_next
            w_prev, dw_prev = w_next, dw_next
            u_prev, du_prev = u_next, du_next
            obs_prev = obs_next

        # ----- FK + normalization → recon_traj -----
        # Stack along time once to reduce per-step copies
        joints_seq = torch.stack(joints_list, dim=1)
        dq_seq = torch.stack(dq_list, dim=1)
        w_seq = torch.stack(w_list, dim=1)
        dw_seq = torch.stack(dw_list, dim=1)
        p_seq = torch.stack(p_list, dim=1)
        dp_seq = torch.stack(dp_list, dim=1)
        obs_seq = torch.stack(obs_list, dim=1)
        objects_seq = torch.stack(objects_list, dim=1)
        log_sig_seq = torch.stack(log_sig_list, dim=1)
        actions_seq = torch.stack(actions_list, dim=1)

        B, T, d = joints_seq.shape
        joint_flat = joints_seq.reshape(B * T, d)  # [B*T, d]
        pos_flat = self.decoder.fk_model(joint_flat)  # [B*T, links*3]
        agent_traj = pos_flat.view(B, T, -1)  # [B,T, links*3]
        combined = torch.cat([agent_traj, objects_seq], dim=-1)  # [B,T,(links+1)*3]
        comb_resh = combined.view(B, T, -1, 3)

        graph_recon = ((comb_resh - self.pos_mean) / self.pos_std).reshape(B, T, -1)

        # ----- return tuple consistent with your trainer -----
        action_seq = actions_seq
        if self.prior == "GMM":
            return {"traj":
                        {"graph": graph_recon,
                          "ptr": ptr,},
                    "state":
                        {"q": joints_seq,
                         "dq": dq_seq,
                         "p": p_seq,
                         "dp": dp_seq,
                         "w": w_seq,
                         "dw": dw_seq,
                         "u": objects_seq,},
                    "act":
                        {"act": action_seq,
                         "mu": 0,
                         "log_std": 0,
                         "log_sigma": log_sig_seq},
                    "obs":
                        {"obs": obs_seq},
                    "aux":
                        [z, mu, logvar, pi_logits]
                    }
        else:
            return {"traj":
                        {"graph": graph_recon,
                        "ptr": ptr,},
                    "state":
                        {"q": joints_seq,
                         "dq": dq_seq,
                         "p": p_seq,
                         "dp": dp_seq,
                         "w": w_seq,
                         "dw": dw_seq},
                    "act":
                        {"act": action_seq,
                         "mu": 0,
                         "log_std": 0,
                         "log_sigma": log_sig_seq},
                    "obs":
                        {"obs": obs_seq},
                    "aux":
                        [z, mu, logvar]
                    }

    # ---------- Loss (masked, correct normalization) ----------
    def loss(
            self,
            recon_traj,  # [B,T,D] normalized FK positions (prediction)
            log_sigma,  # [B,T,D_log]
            orig_traj,  # [B,T,D] normalized FK positions (target)
            act_mu,  # [B,T,A] teacher actions (unused now)
            action_seq,  # [B,T,A] decoder actions (prediction)
            mu_seq,  # [B,T,d]
            log_std_seq,  # [B,T,d]
            mask,  # [B,T,1] float {0,1}
            *args,  # latent tuples per prior
            beta: float,
            lambda_kinematic: float = 1.0,
            lambda_dynamic: float = 0.2,
            gamma: float = 0.95,  # discount for kinematic term only
            wa1: float = 1e-3,  # L1 coeff for action magnitude
            wa2: float = 1e-3,  # L2 coeff for action magnitude
    ):
        """
        Masked sequence loss:
          - Pose: MSE with per-timestep discount gamma^t, masked and normalized.
          - Action: NO teacher; use magnitude regularizer (wa1*L1 + wa2*L2^2), masked.
          - KL: per-sequence.
        """
        B, T, D = recon_traj.shape
        device = recon_traj.device
        dtype = recon_traj.dtype

        mask_bt = mask.squeeze(-1)  # [B,T], {0,1}

        # -------- Pose reconstruction with discount gamma^t --------
        orig_traj = orig_traj.view(B, T, -1)
        pose_step = ((recon_traj[:,:,:-3] - orig_traj[:,:,:-3]) ** 2).mean(dim=-1)  # [B,T]

        t_idx = torch.arange(T, device=device, dtype=dtype)
        w_t = (gamma ** t_idx).to(dtype=dtype)  # [T]
        weighted = pose_step * mask_bt * w_t.unsqueeze(0)  # [B,T]
        denom = (mask_bt * w_t.unsqueeze(0)).sum().clamp_min(1e-6)
        pose_loss = weighted.sum() / denom
        pose_loss = lambda_kinematic * pose_loss

        # -------- Action magnitude regularizer (replace old MSE) --------
        # Per-step mean over action dims
        l1 = action_seq.abs().mean(dim=-1)  # [B,T]
        l2 = (action_seq ** 2).mean(dim=-1)  # [B,T]
        act_reg_step = wa1 * l1 + wa2 * l2  # [B,T]
        valid = mask_bt.sum().clamp_min(1.0)
        action_loss = (act_reg_step * mask_bt).sum().div(valid)
        action_loss = lambda_dynamic * action_loss

        total_recon = pose_loss + action_loss

        # -------- KL (unchanged) --------
        if self.prior == "GMM":
            mu, logvar, pi_logits = args
            Bk, K, Dk = mu.shape
            pi = F.softmax(pi_logits, dim=-1)  # [B,K]

            prior_mu = self.prior_mu.unsqueeze(0).expand(Bk, -1, -1)  # [B,K,D]
            prior_logv = self.prior_logvar.unsqueeze(0).expand(Bk, -1, -1)  # [B,K,D]

            kl_gauss = 0.5 * torch.sum(
                prior_logv - logvar - 1.0
                + (logvar - prior_logv).exp()
                + (mu - prior_mu).pow(2) / prior_logv.exp(),
                dim=-1,
            )  # [B,K]

            log_q = torch.log(pi + 1e-10)
            log_p = math.log(1.0 / K)
            kl_cat = torch.sum(pi * (log_q - log_p), dim=-1)  # [B]

            kl_loss = torch.sum(pi * kl_gauss, dim=-1) + kl_cat
            kl_loss = kl_loss.mean()
        else:
            z, mu, logvar = args
            kl_per = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_loss = kl_per.mean()

        vae_loss = total_recon + beta * kl_loss
        return vae_loss, pose_loss, action_loss, kl_loss
