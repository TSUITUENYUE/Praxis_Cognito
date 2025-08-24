# go2_env_mjx.py
import math
import torch
from typing import List
import jax, jax.numpy as jnp
import jax.dlpack as jdl
import torch.utils.dlpack as tdl
import mujoco
import mujoco.mjx as mjx

# reuse your quaternion helpers (pure torch ops)
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def _rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def _to_j(x_t: torch.Tensor):
    return jdl.from_dlpack(tdl.to_dlpack(x_t.contiguous()))


def _to_t(x_j: jnp.ndarray, like: torch.Tensor):
    t = tdl.from_dlpack(jdl.to_dlpack(x_j))
    return t.to(device=like.device, dtype=like.dtype)


# -------------- PyTorch <-> JAX autograd bridge --------------
class _MJXRollFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qpos_in, qvel_in, qdprev, qdcurr, env_ref):
        # Torch -> JAX
        qpos_j   = jdl.from_dlpack(tdl.to_dlpack(qpos_in.contiguous()))
        qvel_j   = jdl.from_dlpack(tdl.to_dlpack(qvel_in.contiguous()))
        qdprev_j = jdl.from_dlpack(tdl.to_dlpack(qdprev.contiguous()))
        qdcurr_j = jdl.from_dlpack(tdl.to_dlpack(qdcurr.contiguous()))

        # Differentiable batched rollout
        f = env_ref._f_batched  # pure JAX fn: (B,nq),(B,nv),(B,d),(B,d) -> (B,nq),(B,nv)
        (qpos_out_j, qvel_out_j), pullback = jax.vjp(f, qpos_j, qvel_j, qdprev_j, qdcurr_j, has_aux=False)

        # Save pullback closure for backward
        ctx.pullback = pullback
        ctx.like = qpos_in  # for device/dtype on return

        # JAX -> Torch
        qpos_out = tdl.from_dlpack(jdl.to_dlpack(qpos_out_j)).to(qpos_in.dtype)
        qvel_out = tdl.from_dlpack(jdl.to_dlpack(qvel_out_j)).to(qvel_in.dtype)
        return qpos_out, qvel_out

    @staticmethod
    def backward(ctx, g_qpos_out, g_qvel_out):
        # Torch grad -> JAX cotangents
        gqpos_j = jdl.from_dlpack(tdl.to_dlpack(g_qpos_out.contiguous()))
        gqvel_j = jdl.from_dlpack(tdl.to_dlpack(g_qvel_out.contiguous()))

        # Pull back through MJX rollout
        g_qpos_in_j, g_qvel_in_j, g_qdprev_j, g_qdcurr_j = ctx.pullback((gqpos_j, gqvel_j))

        # JAX -> Torch grads
        like = ctx.like
        g_qpos_in = tdl.from_dlpack(jdl.to_dlpack(g_qpos_in_j)).to(like)
        g_qvel_in = tdl.from_dlpack(jdl.to_dlpack(g_qvel_in_j)).to(like)
        g_qdprev  = tdl.from_dlpack(jdl.to_dlpack(g_qdprev_j)).to(like)
        g_qdcurr  = tdl.from_dlpack(jdl.to_dlpack(g_qdcurr_j)).to(like)

        # No grad for env_ref
        return g_qpos_in, g_qvel_in, g_qdprev, g_qdcurr, None


class _RobotView:
    """Thin view mirroring the subset of Genesis robot API your code uses."""
    def __init__(self, env: "Go2EnvMJX"):
        self._e = env

    # ---------- control ----------
    def control_dofs_position(self, target_qpos: torch.Tensor, dofs_idx_local: List[int]):
        # Store desired joint positions for PD; applied in MJX stepping.
        self._e._q_des = target_qpos  # [B, d]

    # ---------- setters used in reset ----------
    def set_dofs_position(self, position: torch.Tensor, dofs_idx_local: List[int],
                          zero_velocity: bool = True, envs_idx: torch.Tensor = None):
        e = self._e
        if envs_idx is None or envs_idx.numel() == 0:
            return
        for i, qa in enumerate(e._j_qpos_idx):
            e._qpos[envs_idx, qa] = position[:, i]
            e._q[envs_idx, i] = position[:, i]
        if zero_velocity:
            for i, va in enumerate(e._j_qvel_idx):
                e._qvel[envs_idx, va] = 0.0
                e._dq[envs_idx, i] = 0.0

    def set_pos(self, pos: torch.Tensor, zero_velocity: bool = False, envs_idx: torch.Tensor = None):
        e = self._e
        if envs_idx is None or envs_idx.numel() == 0:
            return
        bq = e._base_qpos_adr
        e._qpos[envs_idx, bq+0:bq+3] = pos
        e._p[envs_idx] = pos
        if zero_velocity:
            bv = e._base_qvel_adr
            e._qvel[envs_idx, bv+3:bv+6] = 0.0
            e._dp[envs_idx] = 0.0

    def set_quat(self, quat: torch.Tensor, zero_velocity: bool = False, envs_idx: torch.Tensor = None):
        e = self._e
        if envs_idx is None or envs_idx.numel() == 0:
            return
        bq = e._base_qpos_adr
        e._qpos[envs_idx, bq+3:bq+7] = quat  # (w,x,y,z)
        e._w[envs_idx] = quat
        if zero_velocity:
            bv = e._base_qvel_adr
            e._qvel[envs_idx, bv+0:bv+3] = 0.0
            e._dw[envs_idx] = 0.0

    def zero_all_dofs_velocity(self, envs_idx: torch.Tensor = None):
        e = self._e
        if envs_idx is None or envs_idx.numel() == 0:
            return
        e._qvel[envs_idx, :] = 0.0
        e._dq[envs_idx, :] = 0.0
        bv = e._base_qvel_adr
        e._qvel[envs_idx, bv+0:bv+3] = 0.0  # ang
        e._qvel[envs_idx, bv+3:bv+6] = 0.0  # lin
        e._dw[envs_idx] = 0.0
        e._dp[envs_idx] = 0.0

    # ---------- getters ----------
    def get_dofs_position(self, dofs_idx_local: List[int] = None):
        return self._e._q.clone()

    def get_dofs_velocity(self, dofs_idx_local: List[int] = None):
        return self._e._dq.clone()

    def get_pos(self):
        return self._e._p.clone()

    def get_quat(self):
        return self._e._w.clone()  # (w,x,y,z)

    def get_vel(self):
        return self._e._dp.clone()  # WORLD linear

    def get_ang(self):
        return self._e._dw.clone()  # WORLD angular


class _BallView:
    def __init__(self, env: "Go2EnvMJX"):
        self._e = env

    def get_pos(self):
        return self._e._u.clone()

    def get_vel(self):
        return self._e._du.clone()


class Go2EnvMJX:
    """
    MJX version of your Genesis Go2Env using Option B1:
      - Parse robot URDF -> mjSpec
      - Programmatically add plane + free-joint ball to a world spec
      - Attach robot subtree
      - Compile once, then step like Genesis
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, agent, show_viewer=False):
        self.num_envs = int(num_envs)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # -------- derive dims (exactly like your code) --------
        self.num_actions = env_cfg.get("num_actions", 12)
        self.num_commands = command_cfg.get("num_commands", 3)
        derived_num_obs = 3 + 3 + self.num_commands + (3 * self.num_actions) + 3 + 3
        self.num_obs = int(obs_cfg.get("num_obs") or derived_num_obs)
        self.num_privileged_obs = None

        # -------- timing / episode --------
        self.simulate_action_latency = bool(env_cfg.get("simulate_action_latency", True))
        self.dt = float(env_cfg.get("dt", 0.02))
        self.substeps = int(env_cfg.get("substeps", 2))
        self.max_episode_length = math.ceil(env_cfg.get("episode_length_s", 20.0) / self.dt)

        # -------- scales --------
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"].copy()

        # -------- device / dtypes --------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tc_float = torch.float32
        tc_int = torch.int32

        # ======================================================
        # ==========  Build world with mjSpec  =================
        # ======================================================
        world_spec = mujoco.mj_makeSpec()
        world_spec.opt.timestep = self.dt / max(1, self.substeps)

        world_body = mujoco.mjs_findBody(world_spec, "world")
        plane = mujoco.mjs_addGeom(world_body, None)
        plane.type = mujoco.mjtGeom.mjGEOM_PLANE
        plane.size[0] = 0.0
        plane.size[1] = 0.0
        plane.size[2] = 1.0

        ball_b = mujoco.mjs_addBody(world_body, None)
        ball_b.name = "ball"
        ball_j = mujoco.mjs_addJoint(ball_b, None)
        ball_j.type = mujoco.mjtJoint.mjJNT_FREE
        ball_g = mujoco.mjs_addGeom(ball_b, None)
        ball_g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        ball_g.size[0] = float(env_cfg.get("ball_radius", 0.05))

        robot_urdf = env_cfg.get("robot_urdf", None) or agent.urdf
        err = bytearray(4096)
        robot_spec = mujoco.mj_parseXML(robot_urdf, None, err, len(err))
        if robot_spec is None:
            raise RuntimeError(f"URDF parse failed: {err.decode('utf-8', errors='ignore')}")

        frame_el = mujoco.mjs_addFrame(world_body, None).element
        robot_root_el = mujoco.mjs_findBody(robot_spec, "world").element
        mujoco.mjs_attach(frame_el, robot_root_el, "robot-", "")

        m = mujoco.mj_compile(world_spec, None)
        d0 = mujoco.mj_makeData(m)
        self._model = mjx.put_model(m)
        self._data0 = mjx.put_data(m, d0)
        self._m = m

        free_joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free_joints) >= 2, "World must include a free base and a free ball."

        ball_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_free = None
        base_free = None
        for j in free_joints:
            if m.jnt_bodyid[j] == ball_body_id:
                ball_free = j
            else:
                base_free = j
        assert ball_free is not None and base_free is not None, "Could not disambiguate base vs ball free joints."
        self._base_j = int(base_free)

        # --- Agent wiring (matches your code) ---
        joint_names = list(agent.joint_name)
        self.num_actions = len(joint_names)
        self.default_dof_pos = agent.init_angles.to(self.device).to(dtype=tc_float)

        jids = []
        for name in joint_names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert j != -1, f"Joint '{name}' not found in combined model."
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, f"Joint '{name}' must be hinge."
            jids.append(int(j))

        self._base_qpos_adr = int(m.jnt_qposadr[self._base_j])
        self._base_qvel_adr = int(m.jnt_dofadr[self._base_j])
        self._ball_qpos_adr = int(m.jnt_qposadr[ball_free])
        self._ball_qvel_adr = int(m.jnt_dofadr[ball_free])

        self._j_qpos_idx = [int(m.jnt_qposadr[j]) for j in jids]
        self._j_qvel_idx = [int(m.jnt_dofadr[j]) for j in jids]
        self._j_qpos_adr_j = jnp.array(self._j_qpos_idx, dtype=jnp.int32)
        self._j_qvel_adr_j = jnp.array(self._j_qvel_idx, dtype=jnp.int32)

        # ---------- PD gains ----------
        kp_cfg = env_cfg["kp"]; kd_cfg = env_cfg["kd"]
        if isinstance(kp_cfg, (int, float)): kp = torch.full((self.num_actions,), float(kp_cfg), device=self.device, dtype=tc_float)
        else: kp = torch.as_tensor(kp_cfg, device=self.device, dtype=tc_float)
        if isinstance(kd_cfg, (int, float)): kd = torch.full((self.num_actions,), float(kd_cfg), device=self.device, dtype=tc_float)
        else: kd = torch.as_tensor(kd_cfg, device=self.device, dtype=tc_float)
        self._kp_j = jnp.array(kp.cpu().numpy())
        self._kd_j = jnp.array(kd.cpu().numpy())

        # ---------- public sub-objects ----------
        self.robot = _RobotView(self)
        self.ball  = _BallView(self)

        # ---------- buffers (names & shapes match your Genesis env) ----------
        B, d = self.num_envs, self.num_actions
        nq, nv = int(m.nq), int(m.nv)
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device, dtype=tc_float)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device, dtype=tc_float)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.actions = torch.zeros((B, d), device=self.device, dtype=tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((B, d), device=self.device, dtype=tc_float)
        self.dof_vel = torch.zeros((B, d), device=self.device, dtype=tc_float)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.base_pos  = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self.base_quat = torch.zeros((B, 4), device=self.device, dtype=tc_float)
        self.base_lin_vel = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self.base_ang_vel = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=tc_float).repeat(B, 1)
        self.projected_gravity = torch.zeros((B, 3), device=self.device, dtype=tc_float)

        self.obs_buf = torch.zeros((B, self.num_obs), device=self.device, dtype=tc_float)
        self.rew_buf = torch.zeros((B,), device=self.device, dtype=tc_float)
        self.reset_buf = torch.ones((B,), device=self.device, dtype=tc_int)
        self.episode_length_buf = torch.zeros((B,), device=self.device, dtype=tc_int)
        self.commands = torch.zeros((B, self.num_commands), device=self.device, dtype=tc_float)
        self.commands_scale = 0
        self.extras = {"observations": {}}
        self.episode_sums = {name: torch.zeros((B,), device=self.device, dtype=tc_float)
                             for name in self.reward_scales.keys()}

        # internal packed state (qpos,qvel)
        self._qpos = torch.zeros((B, nq), device=self.device, dtype=tc_float)
        self._qvel = torch.zeros((B, nv), device=self.device, dtype=tc_float)
        # convenient views
        self._q  = torch.zeros((B, d), device=self.device, dtype=tc_float)
        self._dq = torch.zeros_like(self._q)
        self._p  = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self._w  = torch.zeros((B, 4), device=self.device, dtype=tc_float)
        self._dp = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self._dw = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self._u  = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self._du = torch.zeros((B, 3), device=self.device, dtype=tc_float)
        self._dv = torch.zeros((B, 3), device=self.device, dtype=tc_float)

        # last desired joints (for first substep latency)
        self._q_des = self.default_dof_pos.unsqueeze(0).repeat(B, 1)

        # ---------- JAX kernels (PD + substeps) ----------
        j_qpos = self._j_qpos_adr_j
        j_qvel = self._j_qvel_adr_j
        kp_j = self._kp_j
        kd_j = self._kd_j

        @jax.jit
        def _pd_tau(qpos, qvel, q_des):
            qj  = qpos.at[j_qpos].get()
            dqj = qvel.at[j_qvel].get()
            return kp_j * (q_des - qj) + kd_j * (0.0 - dqj)

        @jax.jit
        def _one_step(model, data, q_des):
            tau = _pd_tau(data.qpos, data.qvel, q_des)            # [d]
            qfrc = jnp.zeros_like(data.qvel).at[j_qvel].set(tau)  # torques at joint DOFs
            return mjx.step(model, data.replace(qfrc_applied=qfrc))

        def _roll(model, data, q_des_prev, q_des_curr, substeps, use_latency: bool):
            def body(k, d):
                qd = jax.lax.select((k == 0) & use_latency, q_des_prev, q_des_curr)
                return _one_step(model, d, qd)
            return jax.lax.fori_loop(0, substeps, body, data)

        self._roll = jax.jit(_roll, static_argnames=("substeps", "use_latency"))

        # -------- Pure, batched JAX function for VJP (used by autograd bridge) --------
        def _f_batched(qpos, qvel, qdprev, qdcurr):
            def _step_one(qp, qv, qd0, qd1):
                d_in = self._data0.replace(qpos=qp, qvel=qv)
                d_out = self._roll(self._model, d_in, qd0, qd1, self.substeps, bool(self.simulate_action_latency))
                return d_out.qpos, d_out.qvel
            return jax.vmap(_step_one)(qpos, qvel, qdprev, qdcurr)
        self._f_batched = jax.jit(_f_batched)

        # ---------- initialize like your reset() ----------
        self.reset()

    # ----------------- command sampling (unchanged) -----------------
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = _rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = _rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = _rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    # ----------------- observation refresh -----------------
    def _refresh_obs(self):
        inv_base_quat = inv_quat(self.base_quat)
        robot_vel_w = self.robot.get_vel()
        self.base_lin_vel[:] = transform_by_quat(robot_vel_w, inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        # use latest views
        self.dof_pos[:] = self.robot.get_dofs_position()
        self.dof_vel[:] = self.robot.get_dofs_velocity()

        ball_pos = self.ball.get_pos()
        ball_vel = self.ball.get_vel()
        self.relative_ball_pos = transform_by_quat(ball_pos - self.base_pos, inv_base_quat)
        rel_vel_w = ball_vel - robot_vel_w
        self.relative_ball_vel = transform_by_quat(rel_vel_w, inv_base_quat)

        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                exec_actions,
                self.relative_ball_pos,
                self.relative_ball_vel,
            ],
            dim=-1,
        )
        self.extras["observations"]["critic"] = self.obs_buf

    # ----------------- step (now differentiable) -----------------
    def step(self, actions: torch.Tensor):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local=list(range(self.num_actions)))

        # Differentiable MJX rollout via PyTorch autograd.Function
        qpos_out, qvel_out = _MJXRollFn.apply(
            self._qpos,
            self._qvel,
            self._q_des,
            (self.actions * self.env_cfg["action_scale"] + self.default_dof_pos),
            self
        )
        self._qpos = qpos_out
        self._qvel = qvel_out

        # update buffers (exactly like your code)
        self.episode_length_buf += 1

        bq = self._base_qpos_adr; bv = self._base_qvel_adr
        self.base_pos[:]  = self._qpos[:, bq+0:bq+3]
        self.base_quat[:] = self._qpos[:, bq+3:bq+7]
        # keep internal views in sync
        self._p[:] = self.base_pos
        self._w[:] = self.base_quat

        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * inv_quat(self.base_init_quat), self.base_quat),
        )
        inv_base_quat = inv_quat(self.base_quat)
        # WORLD velocities pulled directly from qvel
        world_lin = self._qvel[:, bv+3:bv+6]
        world_ang = self._qvel[:, bv+0:bv+3]
        self.base_lin_vel[:] = transform_by_quat(world_lin, inv_base_quat)  # WORLD->body
        self.base_ang_vel[:] = transform_by_quat(world_ang, inv_base_quat)
        self._dp[:] = world_lin
        self._dw[:] = world_ang

        # joints (also mirror to internal views)
        for i, (qa, va) in enumerate(zip(self._j_qpos_idx, self._j_qvel_idx)):
            self.dof_pos[:, i] = self._qpos[:, qa]
            self.dof_vel[:, i] = self._qvel[:, va]
        self._q[:]  = self.dof_pos
        self._dq[:] = self.dof_vel

        # ball
        uq = self._ball_qpos_adr; uv = self._ball_qvel_adr
        ball_pos = self._qpos[:, uq+0:uq+3]
        ball_vel_lin = self._qvel[:, uv+3:uv+6]
        robot_vel_w = world_lin
        self._u[:] = ball_pos
        self._du[:] = ball_vel_lin

        self.relative_ball_pos = transform_by_quat(ball_pos - self.base_pos, inv_base_quat)
        self.relative_ball_vel = transform_by_quat(ball_vel_lin - robot_vel_w, inv_base_quat)

        # termination & reset
        self.reset_buf = (self.episode_length_buf > self.max_episode_length)
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=self.obs_buf.dtype)
        if time_out_idx.numel() > 0:
            self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # rewards (kept same structure; adjust scaling if you pre-multiply in init)
        self.rew_buf[:] = 0.0
        for name, scale in self.reward_scales.items():
            fn = getattr(self, "_reward_" + name, None)
            if fn is None: continue
            rew = fn() * (scale * self.dt)
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # observations (same layout/order)
        exec_actions_now = self.last_actions if self.simulate_action_latency else self.actions
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],
                self.base_ang_vel * self.obs_scales["ang_vel"],
                transform_by_quat(self.global_gravity, inv_base_quat),
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                exec_actions_now,
                self.relative_ball_pos,
                self.relative_ball_vel,
            ],
            dim=-1,
        )
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ----------------- public accessors -----------------
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ----------------- resets (mirror Genesis) -----------------
    def reset_with_ball_rel(self, u_rel, du_rel):
        u_w = transform_by_quat(u_rel, self.base_init_quat) + self.base_init_pos
        du_w = transform_by_quat(du_rel, self.base_init_quat)

        uq = self._ball_qpos_adr; uv = self._ball_qvel_adr
        self._qpos[:, uq+0:uq+3] = u_w
        self._qpos[:, uq+3:uq+7] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                device=self.device, dtype=self._qpos.dtype).expand(self.num_envs, 4)
        self._qvel[:, uv+3:uv+6] = du_w   # linear
        self._qvel[:, uv+0:uv+3] = 0.0    # angular

        # sync views
        self._u[:] = self._qpos[:, uq+0:uq+3]
        self._du[:] = self._qvel[:, uv+3:uv+6]
        self._refresh_obs()

    def reset_idx(self, envs_idx):
        if envs_idx.numel() == 0:
            return
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=list(range(self.num_actions)),
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # sync internal views for these envs
        self._q[envs_idx]  = self.dof_pos[envs_idx]
        self._dq[envs_idx] = self.dof_vel[envs_idx]
        self._p[envs_idx]  = self.base_pos[envs_idx]
        self._w[envs_idx]  = self.base_quat[envs_idx]
        self._dp[envs_idx] = 0.0
        self._dw[envs_idx] = 0.0
        uq = self._ball_qpos_adr; uv = self._ball_qvel_adr
        self._u[envs_idx]  = self._qpos[envs_idx, uq+0:uq+3]
        self._du[envs_idx] = self._qvel[envs_idx, uv+3:uv+6]

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # episode stats
        self.extras["episode"] = {}
        for key in self.reward_scales.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        bq = self._base_qpos_adr
        self._qpos[:, bq+0:bq+3] = self.base_init_pos.reshape(1, 3).expand(self.num_envs, 3)
        self._qpos[:, bq+3:bq+7] = self.base_init_quat.reshape(1, 4).expand(self.num_envs, 4)

        uq = self._ball_qpos_adr
        self._qpos[:, uq+3:uq+7] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                device=self.device, dtype=self._qpos.dtype).expand(self.num_envs, 4)

        self._qvel.zero_()

        # mirror to convenience views
        self.base_pos[:] = self._qpos[:, bq+0:bq+3]
        self.base_quat[:] = self._qpos[:, bq+3:bq+7]
        self._p[:] = self.base_pos
        self._w[:] = self.base_quat
        for i, (qa, va) in enumerate(zip(self._j_qpos_idx, self._j_qvel_idx)):
            self.dof_pos[:, i] = self._qpos[:, qa]
            self.dof_vel[:, i] = self._qvel[:, va]
        self._q[:]  = self.dof_pos
        self._dq[:] = self.dof_vel
        uq = self._ball_qpos_adr; uv = self._ball_qvel_adr
        self._u[:]  = self._qpos[:, uq+0:uq+3]
        self._du[:] = self._qvel[:, uv+3:uv+6]
        self._dp.zero_()
        self._dw.zero_()

        self._refresh_obs()
        return self.obs_buf, self.extras

    # ----------------- reward functions (same as your file) -----------------
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_survive(self):
        return torch.ones_like(self.rew_buf)

    def _reward_termination(self):
        failures = torch.logical_and(self.reset_buf, ~self.extras["time_outs"].bool())
        return -failures.float()
