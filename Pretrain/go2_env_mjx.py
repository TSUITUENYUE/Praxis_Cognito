# go2_env_mjx.py (MJX + PyTorch-grad via custom autograd.Function)
import os
import io
import math
import tempfile
from typing import List
import copy
import torch

# ---- PATCH: cap JAX/XLA GPU preallocation before importing jax ----
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# ---------------------------------------------------------------

import jax
import jax.numpy as jnp
import jax.dlpack as jdl
import torch.utils.dlpack as tdl
import mujoco
import mujoco.mjx as mjx
import xml.etree.ElementTree as ET
import re
import uuid

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def _rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def _to_j(x_t: torch.Tensor):
    return jdl.from_dlpack(tdl.to_dlpack(x_t.contiguous()))


def _to_t(x_j: jnp.ndarray, like: torch.Tensor):
    t = tdl.from_dlpack(jdl.to_dlpack(x_j))
    # ---- PATCH: avoid unnecessary copies ----
    if t.dtype != like.dtype or t.device != like.device:
        t = t.to(device=like.device, dtype=like.dtype)
    return t
    # -----------------------------------------


def _strip_visual_meshes(urdf_path: str) -> str:
    """Fallback if package:// meshes cause issues: drop all <visual> blocks."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall(".//link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    fd, stripped = tempfile.mkstemp(suffix="_no_visuals.urdf")
    os.close(fd)
    tree.write(stripped, encoding="utf-8", xml_declaration=True)
    return stripped

def _compile_world_with_urdf(robot_xml_path: str, dt: float, substeps: int, ball_radius: float):
    robot_xml_path = os.path.abspath(robot_xml_path)
    base_dir = os.path.dirname(robot_xml_path)

    rroot = ET.parse(robot_xml_path).getroot()

    pre_sections  = ["compiler", "size", "visual", "statistic", "default", "asset"]
    post_sections = ["sensor", "tendon", "actuator", "contact", "equality", "custom"]  # we'll handle keyframe separately

    # ---- new root ----
    wroot = ET.Element("mujoco", {"model": "go2_world"})

    # option: keep robot's, override timestep
    ropt = rroot.find("option")
    opt_attrs = dict(ropt.attrib) if ropt is not None else {}
    opt_attrs["timestep"] = f"{dt / max(1, substeps):.9f}"
    ET.SubElement(wroot, "option", opt_attrs)

    # pre-worldbody
    for name in pre_sections:
        sec = rroot.find(name)
        if sec is not None:
            wroot.append(copy.deepcopy(sec))

    # worldbody
    w_world = ET.SubElement(wroot, "worldbody")

    # bring in robot bodies first (to keep original joint order first in qpos)
    r_worldbody = rroot.find("worldbody")
    if r_worldbody is not None:
        for child in list(r_worldbody):
            w_world.append(copy.deepcopy(child))

    # floor
    ET.SubElement(w_world, "geom", {
        "name": "floor", "type": "plane", "size": "100 100 0.1", "rgba": "0.8 0.9 1 1"
    })

    # ball LAST (so its freejoint qpos comes AFTER robot’s qpos)
    r = max(float(ball_radius), 1e-8)
    b = ET.SubElement(w_world, "body", {"name": "ball", "pos": "0 0 0.1"})
    ET.SubElement(b, "freejoint")
    ET.SubElement(b, "geom", {"name":"ball_geom", "type":"sphere", "size":f"{r:.6f}", "rgba":"1 0.3 0.3 1"})

    # post-worldbody (except keyframes for now)
    for name in post_sections:
        for sec in rroot.findall(name):
            wroot.append(copy.deepcopy(sec))

    # --- keyframes: patch qpos length by appending ball qpos (pos3+quat4) ---
    r_keyframe = rroot.find("keyframe")
    if r_keyframe is not None:
        kcopy = copy.deepcopy(r_keyframe)
        for key in kcopy.findall("key"):
            if "qpos" in key.attrib:
                vals = key.attrib["qpos"].split()
                # append 7 for the ball freejoint (pos xyz + quat wxyz)
                vals += ["0", "0", "0.1", "1", "0", "0", "0"]
                key.set("qpos", " ".join(vals))
        wroot.append(kcopy)

    # write next to robot so mesh paths resolve
    world_xml_path = os.path.join(base_dir, "_go2_world.xml")
    ET.ElementTree(wroot).write(world_xml_path, encoding="utf-8", xml_declaration=True)

    # compile
    model = mujoco.MjModel.from_xml_path(world_xml_path)
    return model, world_xml_path, robot_xml_path


def _ensure_free_root(mjcf_path: str, root_body_name: str = "root"):
    """Inject <freejoint/> under the robot root body if it's missing."""
    with open(mjcf_path, "r", encoding="utf-8") as f:
        xml = f.read()
    # Find opening tag of the root body
    m = re.search(rf'<body\b[^>]*\bname\s*=\s*"{re.escape(root_body_name)}"[^>]*>', xml)
    if not m:
        print("not")
        return
    insert_at = m.end()
    # Already has a free joint near the start of this body?
    window = xml[insert_at:insert_at+4000]
    if re.search(r'<freejoint\s*/?>', window) or re.search(r'<joint[^>]+type\s*=\s*"free"', window):
        return
    xml = xml[:insert_at] + "\n    <freejoint/>\n" + xml[insert_at:]
    with open(mjcf_path, "w", encoding="utf-8") as f:
        f.write(xml)


def _enable_only_mjx_supported_contacts(m: mujoco.MjModel):
    """
    Keep only contacts MJX supports well: plane/sphere/capsule.
    - Convert cylinders -> capsules (size mapping is compatible).
    - Disable collisions for boxes/meshes/ellipsoids/etc via contype/conaffinity=0.
    """
    # 1) Convert cylinders to capsules (keeps radius=size[0], half-length=size[1])
    for gi in range(m.ngeom):
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            m.geom_type[gi] = mujoco.mjtGeom.mjGEOM_CAPSULE

    # 2) Disable all collisions by default
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    # 3) Re-enable only plane/sphere/capsule
    for gi in range(m.ngeom):
        if m.geom_type[gi] in (
            mujoco.mjtGeom.mjGEOM_PLANE,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
        ):
            m.geom_contype[gi] = 1
            m.geom_conaffinity[gi] = 1


# ------------------------------ PyTorch <-> MJX autograd bridge ------------------------------

class _MJXStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                qpos_in: torch.Tensor,
                qvel_in: torch.Tensor,
                qd_prev: torch.Tensor,
                qd_curr: torch.Tensor,
                env: "Go2EnvMJX"):
        # Truncate BPTT: treat state-like inputs as constants
        qpos_d   = qpos_in.detach()
        qvel_d   = qvel_in.detach()
        qdprev_d = qd_prev.detach()

        # Remember env; save only the action path tensor for autograd bookkeeping
        ctx.env = env
        ctx.save_for_backward(qd_curr)
        ctx._cached_inputs = (qpos_d, qvel_d, qdprev_d)

        # JAX arrays
        qp_j     = _to_j(qpos_d)
        qv_j     = _to_j(qvel_d)
        qdprev_j = _to_j(qdprev_d)
        qdcurr_j = _to_j(qd_curr)

        # Shape constants
        nq = int(env._m.nq)
        nv = int(env._m.nv)
        ctx._nq = nq
        ctx._nv = nv

        # Define a flat function at this primal point
        def f_flat(qdcurr):
            qpos_j, qvel_j = env._batched_step(
                env._model, env._data0,
                qp_j, qv_j, qdprev_j, qdcurr,
                env.substeps, bool(env.simulate_action_latency)
            )
            return jnp.concatenate([qpos_j, qvel_j], axis=-1)  # [B, nq+nv]

        # Linearize once at the primal; cache linear map for VJP
        y_flat, f_lin = jax.linearize(f_flat, qdcurr_j)       # y_flat: [B, nq+nv]
        ctx._f_lin = f_lin
        ctx._qdcurr_ex = qdcurr_j  # example array for transpose shape inference

        # Split outputs and convert to torch
        qpos_out_j = y_flat[..., :nq]
        qvel_out_j = y_flat[..., nq:]
        qpos_out_t = _to_t(qpos_out_j, qpos_in)
        qvel_out_t = _to_t(qvel_out_j, qvel_in)
        return qpos_out_t, qvel_out_t

    @staticmethod
    def backward(ctx, g_qpos_out: torch.Tensor, g_qvel_out: torch.Tensor):
        (qd_curr,) = ctx.saved_tensors
        env = ctx.env
        qpos_d, qvel_d, qdprev_d = ctx._cached_inputs

        # 0) Guard incoming PyTorch grads
        g_qpos_out = torch.nan_to_num(g_qpos_out, 0.0, 0.0, 0.0)
        g_qvel_out = torch.nan_to_num(g_qvel_out, 0.0, 0.0, 0.0)

        # 1) JAX views
        g_qp_j = _to_j(g_qpos_out.contiguous())
        g_qv_j = _to_j(g_qvel_out.contiguous())
        qdcurr_j = _to_j(qd_curr)

        # 2) Build flat cotangent
        g_out = jnp.concatenate([g_qp_j, g_qv_j], axis=-1).astype(qdcurr_j.dtype)

        # 3) Apply transpose of cached linear map (JVP->VJP)
        try:
            lt = jax.linear_transpose(ctx._f_lin, qdcurr_j)  # returns a callable
            (g_qdcurr_j,) = lt(g_out)
            g_qdcurr_j = jnp.nan_to_num(g_qdcurr_j, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            # Safe fallback: zero grad
            g_qdcurr_j = jnp.zeros_like(qdcurr_j)

        # 4) Back to torch
        g_qd_curr = _to_t(g_qdcurr_j, qd_curr)
        g_qd_curr = torch.nan_to_num(g_qd_curr, 0.0, 0.0, 0.0)

        # No grads to state-like inputs (by design)
        g_qpos_in = torch.zeros_like(qpos_d)
        g_qvel_in = torch.zeros_like(qvel_d)
        g_qd_prev = torch.zeros_like(qdprev_d)

        return g_qpos_in, g_qvel_in, g_qd_prev, g_qd_curr, None


# ------------------------------------ Thin API views ------------------------------------

class _RobotView:
    """Thin view mirroring the subset of Genesis robot API your code uses."""
    def __init__(self, env: "Go2EnvMJX"):
        self._e = env

    # ---------- control ----------
    def control_dofs_position(self, target_qpos: torch.Tensor, dofs_idx_local: List[int]):
        # in control_dofs_position(...)
        self._e._q_des = target_qpos.detach().clone()

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


# --------------------------------------- Env ---------------------------------------

class Go2EnvMJX:
    """
    MJX version of your Genesis Go2Env using a temporary MJCF world:
      - Load robot URDF → canonical MJCF via mj_saveLastXML
      - Build MJCF world (plane + free ball) including the robot
      - Compile once, step like Genesis
      - Optional differentiable step via JAX VJP wrapped in torch.autograd.Function
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, agent, show_viewer=False):
        self.num_envs = int(num_envs)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.differentiable_step = bool(env_cfg.get("differentiable_step", True))

        # -------- derive dims (exactly like your code) --------
        self.num_actions = env_cfg.get("num_actions", 12)
        self.num_commands = command_cfg.get("num_commands", 3)
        derived_num_obs = 3 + 3 + self.num_commands + (3 * self.num_actions) + 3 + 3
        self.num_obs = int(obs_cfg.get("num_obs") or derived_num_obs)
        self.num_privileged_obs = None

        # -------- timing / episode --------
        self.simulate_action_latency = bool(env_cfg.get("simulate_action_latency", True))
        self.dt = float(env_cfg.get("dt", 0.02))
        self.substeps = int(env_cfg.get("substeps", 1))
        self.max_episode_length = math.ceil(env_cfg.get("episode_length_s", 20.0) / self.dt)

        # -------- scales --------
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"].copy()

        # -------- device / dtypes --------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tc_float = torch.float32
        tc_int = torch.int32

        # ===== Build world by compiling MJCF with included URDF =====
        robot_urdf = agent.urdf
        ball_radius = float(env_cfg.get("ball_radius", 0.05))
        m, world_path, robot_mjcf_path = _compile_world_with_urdf(
            robot_xml_path="Pretrain/urdfs/go2/go2.xml", dt=self.dt, substeps=self.substeps, ball_radius=ball_radius
        )
        # PATCH: restrict contacts before handing model to MJX
        _enable_only_mjx_supported_contacts(m)

        d0 = mujoco.MjData(m)
        self._model = mjx.put_model(m)

        self._data0 = mjx.put_data(m, d0)
        self._m = m

        # locate indices (free base, hinges by name, free ball)
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

        joint_names = list(agent.joint_name)
        self.num_actions = len(joint_names)
        self.default_dof_pos = agent.init_angles.to(self.device).to(dtype=tc_float)

        jids = []
        for name in joint_names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert j != -1, f"Joint '{name}' not found in combined model."
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, f"Joint '{name}' must be hinge."
            jids.append(int(j))

        # addresses
        self._base_qpos_adr = int(m.jnt_qposadr[self._base_j])   # pos3+quat4
        self._base_qvel_adr = int(m.jnt_dofadr[self._base_j])    # ang3+lin3
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
        self.reset_buf = torch.ones((B,), device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros((B,), device=self.device, dtype=torch.int32)
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

        # ---- PATCH: preallocate time_outs buffer to avoid per-step alloc ----
        self._time_outs_buf = torch.zeros((B,), device=self.device, dtype=self.obs_buf.dtype)
        # --------------------------------------------------------------------

        # ---------- JAX kernels (PD + substeps) ----------
        j_qpos = self._j_qpos_adr_j
        j_qvel = self._j_qvel_adr_j
        kp_j = self._kp_j
        kd_j = self._kd_j

        # ---------- JAX kernels (PD + substeps) ----------
        j_qpos = self._j_qpos_adr_j
        j_qvel = self._j_qvel_adr_j
        kp_j = self._kp_j
        kd_j = self._kd_j

        # ---------- JAX kernels (PD + substeps) ----------
        j_qpos = self._j_qpos_adr_j
        j_qvel = self._j_qvel_adr_j
        kp_j = self._kp_j
        kd_j = self._kd_j

        @jax.jit
        def _pd_tau(qpos, qvel, q_des):
            # keep dtype stable
            q_des = q_des.astype(qpos.dtype)
            qj = qpos.at[j_qpos].get()
            dqj = qvel.at[j_qvel].get()
            return kp_j * (q_des - qj) + kd_j * (0.0 - dqj)

        @jax.jit
        def _one_step(model, data, q_des):
            tau = _pd_tau(data.qpos, data.qvel, q_des)
            qfrc = jnp.zeros_like(data.qvel).at[j_qvel].set(tau)  # [nv]
            return mjx.step(model, data.replace(qfrc_applied=qfrc))

        def _roll(model, data, q_des_prev, q_des_curr, substeps, use_latency: bool):
            # convert python bool → JAX scalar bool
            use_lat = jnp.asarray(use_latency, dtype=bool)

            def body(d, k):
                # scalar bool
                use_prev = jnp.logical_and(k == 0, use_lat)
                qd = jax.lax.select(use_prev, q_des_prev, q_des_curr).astype(d.qpos.dtype)
                d_next = _one_step(model, d, qd)
                # IMPORTANT: y must be an array with fixed dtype/shape, not None
                y = jnp.zeros((), dtype=jnp.int32)
                return d_next, y

            idx = jnp.arange(substeps, dtype=jnp.int32)
            data_out, _ = jax.lax.scan(body, data, idx)
            return data_out

        self._roll = jax.jit(_roll, static_argnames=("substeps", "use_latency"))

        def _batched_step(model, data, qpos, qvel, qdprev, qdcurr, substeps, use_latency: bool):
            def _one(qp, qv, qdpr, qdcr):
                d_in = data.replace(qpos=qp, qvel=qv)
                d_out = _roll(model, d_in, qdpr, qdcr, substeps, use_latency)
                return d_out.qpos, d_out.qvel

            return jax.vmap(_one)(qpos, qvel, qdprev, qdcurr)

        self._batched_step = jax.jit(_batched_step, static_argnames=("substeps", "use_latency"))
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
        # ---- PATCH: avoid keeping GPU graph in extras ----
        self.extras["observations"]["critic"] = self.obs_buf.detach()
        # --------------------------------------------------

    # ----------------- step (Genesis-like) -----------------
    def step(self, actions: torch.Tensor):
        # clip & latency like Genesis
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local=list(range(self.num_actions)))

        # desired joints from current actions (keep graph)
        qd_curr_torch = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # differentiable vs non-diff rollout
        if self.differentiable_step and qd_curr_torch.requires_grad:
            # ---- PATCH: pass detached state to prevent graph growth ----
            qpos_out_t, qvel_out_t = _MJXStep.apply(
                self._qpos.detach(), self._qvel.detach(), self._q_des.detach(), qd_curr_torch, self
            )
            # -------------------------------------------------------------
        else:
            qpos_j = _to_j(self._qpos)
            qvel_j = _to_j(self._qvel)
            qdprev_j = _to_j(self._q_des)
            qdcurr_j = _to_j(qd_curr_torch)

            qpos_out_j, qvel_out_j = self._batched_step(
                self._model, self._data0,
                qpos_j, qvel_j, qdprev_j, qdcurr_j,
                self.substeps, bool(self.simulate_action_latency)
            )
            qpos_out_t = _to_t(qpos_out_j, self._qpos)
            qvel_out_t = _to_t(qvel_out_j, self._qvel)

        # assign (these may carry grad_fn via qd_curr path)
        self._qpos = qpos_out_t
        self._qvel = qvel_out_t

        # ---- update buffers (matches your Genesis logic) ----
        self.episode_length_buf += 1

        bq = self._base_qpos_adr; bv = self._base_qvel_adr
        self.base_pos[:]  = self._qpos[:, bq+0:bq+3]
        self.base_quat[:] = self._qpos[:, bq+3:bq+7]
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * inv_quat(self.base_init_quat), self.base_quat),
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self._qvel[:, bv+3:bv+6], inv_base_quat)  # WORLD->body
        self.base_ang_vel[:] = transform_by_quat(self._qvel[:, bv+0:bv+3], inv_base_quat)

        # joints
        for i, (qa, va) in enumerate(zip(self._j_qpos_idx, self._j_qvel_idx)):
            self.dof_pos[:, i] = self._qpos[:, qa]
            self.dof_vel[:, i] = self._qvel[:, va]

        # ball
        uq = self._ball_qpos_adr; uv = self._ball_qvel_adr
        ball_pos = self._qpos[:, uq+0:uq+3]
        ball_vel_lin = self._qvel[:, uv+3:uv+6]
        robot_vel_w = self._qvel[:, bv+3:bv+6]
        self.relative_ball_pos = transform_by_quat(ball_pos - self.base_pos, inv_base_quat)
        self.relative_ball_vel = transform_by_quat(ball_vel_lin - robot_vel_w, inv_base_quat)

        # termination & reset
        self.reset_buf = (self.episode_length_buf > self.max_episode_length)
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        # ---- PATCH: reuse preallocated buffer instead of zeros_like ----
        self._time_outs_buf.zero_()
        if time_out_idx.numel() > 0:
            self._time_outs_buf[time_out_idx] = 1.0
        self.extras["time_outs"] = self._time_outs_buf
        # ----------------------------------------------------------------

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # rewards (kept same structure)
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
        # 1) When you build obs_buf:
        exec_actions_now = (self.last_actions if self.simulate_action_latency else self.actions)
        exec_actions_now = exec_actions_now.detach().clone()

        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],
                self.base_ang_vel * self.obs_scales["ang_vel"],
                transform_by_quat(self.global_gravity, inv_base_quat),
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                exec_actions_now,  # <-- snapshot, not alias
                self.relative_ball_pos,
                self.relative_ball_vel,
            ],
            dim=-1,
        )

        # 2) When you update "last_*" buffers, avoid in-place writes:
        self.last_actions = self.actions.detach().clone()
        self.last_dof_vel = self.dof_vel.detach().clone()

        # ---- PATCH: avoid keeping GPU graph in extras ----
        self.extras["observations"]["critic"] = self.obs_buf.detach()
        # --------------------------------------------------
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ----------------- public accessors -----------------
    def get_observations(self):
        # ---- PATCH: avoid keeping GPU graph in extras ----
        self.extras["observations"]["critic"] = self.obs_buf.detach()
        # --------------------------------------------------
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
        # base pose in packed arrays
        bq = self._base_qpos_adr
        self._qpos[:, bq+0:bq+3] = self.base_init_pos.reshape(1, 3).expand(self.num_envs, 3)
        self._qpos[:, bq+3:bq+7] = self.base_init_quat.reshape(1, 4).expand(self.num_envs, 4)

        # identity ball quat
        uq = self._ball_qpos_adr
        self._qpos[:, uq+3:uq+7] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                device=self.device, dtype=self._qpos.dtype).expand(self.num_envs, 4)

        # zero velocities
        self._qvel.zero_()

        # mirror to convenience views
        self.base_pos[:] = self._qpos[:, bq+0:bq+3]
        self.base_quat[:] = self._qpos[:, bq+3:bq+7]
        for i, (qa, va) in enumerate(zip(self._j_qpos_idx, self._j_qvel_idx)):
            self.dof_pos[:, i] = self._qpos[:, qa]
            self.dof_vel[:, i] = self._qvel[:, va]

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
