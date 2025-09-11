import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, agent, show_viewer=False):
        self.num_envs = num_envs
        # derive observation dimension if not provided (keeps pipeline consistent)
        self.num_actions = env_cfg.get("num_actions", 12)
        self.num_commands = command_cfg.get("num_commands", 14)
        # base_ang_vel(3) + gravity(3) + commands(C) + dof_pos(A) + dof_vel(A) + actions(A) + rel_ball_pos(3) + rel_ball_vel(3)
        derived_num_obs = 3 + 3 + self.num_commands + (3 * self.num_actions) + 3 + 3
        self.num_obs = int(obs_cfg.get("num_obs") or derived_num_obs)
        self.num_privileged_obs = None
        self.device = gs.device

        # honor config if provided
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = float(env_cfg.get("dt", 0.02))  # control frequency default 50Hz
        self.max_episode_length = math.ceil(env_cfg.get("episode_length_s", 20.0) / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.ee_idx = [9,10,11,12]
        self.contact_on_force_N = float(self.env_cfg.get("contact_on_force_N", 15.0))
        self.contact_off_force_N = float(self.env_cfg.get("contact_off_force_N", 7.5))

        # Buffer to hold sticky flags
        self.contact_flags = torch.zeros((self.num_envs, len(self.ee_idx)), device=gs.device, dtype=gs.tc_float)
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.Plane(fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=agent.urdf,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.05,
            )
        )
        # build
        self.scene.build(n_envs=num_envs)

        # --- Agent wiring (preferred over env_cfg when provided) ---
        joint_names = agent.joint_name
        self.num_actions = len(joint_names)
        self.default_dof_pos = agent.init_angles.to(gs.device).to(dtype=gs.tc_float)

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in joint_names]

        # names to indices

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = 1.0
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # ---- indices (union layout) ----
        # 0-2   locomote:   [vx, vy, wz]
        # 3-5   body pose:  [Δz, Δpitch, Δroll]
        # 6-9   limbvel:    [ee_vx, ee_vy, ee_vz, imp_s]
        # 10-12 contact:    [Fn, vt, φ]
        # 13-15 hop:        [apex_h, ψ, pitch_imp]

        # locomote
        # ----- OBJECT-CENTRIC LOCOMOTION (0..2) -----

        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["standoff_r_range"], (len(envs_idx),), gs.device)  # r
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["approach_psi_range"], (len(envs_idx),), gs.device)  # ψ
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["nom_speed_range"], (len(envs_idx),), gs.device)  # v_nom

        # ----- BODY POSE (3..5) UNCHANGED -----
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["dz_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 4] = gs_rand_float(*self.command_cfg["pitch_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 5] = gs_rand_float(*self.command_cfg["roll_range"], (len(envs_idx),), gs.device)

        # body pose
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["dz_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 4] = gs_rand_float(*self.command_cfg["pitch_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 5] = gs_rand_float(*self.command_cfg["roll_range"], (len(envs_idx),), gs.device)

        # limbvel
        self.commands[envs_idx, 6] = gs_rand_float(*self.command_cfg["ee_vx_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 7] = gs_rand_float(*self.command_cfg["ee_vy_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 8] = gs_rand_float(*self.command_cfg["ee_vz_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 9] = gs_rand_float(*self.command_cfg["imp_s_range"], (len(envs_idx),), gs.device)

        # contact hold
        self.commands[envs_idx, 10] = gs_rand_float(*self.command_cfg["Fn_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 11] = gs_rand_float(*self.command_cfg["vt_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 12] = gs_rand_float(*self.command_cfg["phi_range"], (len(envs_idx),), gs.device)

        # hop
        self.commands[envs_idx, 13] = gs_rand_float(*self.command_cfg["apex_h_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 14] = gs_rand_float(*self.command_cfg["psi_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 15] = gs_rand_float(*self.command_cfg["pitch_imp_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        # actions = torch.tanh(actions)
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        ball_pos = self.ball.get_pos()
        ball_vel = self.ball.get_vel()
        # express in base frame
        self.relative_ball_pos = transform_by_quat(ball_pos - self.base_pos, inv_base_quat)
        self.relative_ball_vel = transform_by_quat(ball_vel - self.robot.get_vel(), inv_base_quat)
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.commands * self.commands_scale,  # 16
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # A
                self.dof_vel * self.obs_scales["dof_vel"],  # A
                exec_actions,  # A
                self.relative_ball_pos,  # 3
                self.relative_ball_vel  # 3
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_with_ball_rel(self, u_rel, du_rel):
        u_w = transform_by_quat(u_rel, self.base_init_quat) + self.base_init_pos
        du_w = transform_by_quat(du_rel, self.base_init_quat)
        self.ball.set_pos(u_w)
        self.ball.set_dofs_velocity(du_w, dofs_idx_local=[0, 1, 2])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        ball_pos = torch.zeros(len(envs_idx), 3, device=gs.device)
        ball_pos[:, 0] = gs_rand_float(0.5, 3, (len(envs_idx),), gs.device)
        ball_pos[:, 1] = gs_rand_float(-1.0, 1.0, (len(envs_idx),), gs.device)
        ball_pos[:, 2] = gs_rand_float(0.0, 1.0, (len(envs_idx),), gs.device)
        self.ball.set_pos(ball_pos, envs_idx=envs_idx)

        ball_vel = torch.zeros(len(envs_idx), 3, device=gs.device)
        ball_vel[:, 0] = gs_rand_float(-1.0, -0.5, (len(envs_idx),), gs.device)  # Flying towards robot
        ball_vel[:, 1] = gs_rand_float(-0.5, 0.5, (len(envs_idx),), gs.device)
        ball_vel[:, 2] = gs_rand_float(1.0, 2.0, (len(envs_idx),), gs.device)  # Flying upwards a bit
        self.ball.set_dofs_velocity(ball_vel, dofs_idx_local=[0, 1, 2], envs_idx=envs_idx)

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        return self.obs_buf, self.extras

    # ------------ reward functions----------------
    # ----------------- helpers (internal) -----------------
    def _angle_wrap(self, x):
        return (x + math.pi) % (2 * math.pi) - math.pi

    # Returns (N, L, 3) foot linear velocities in BODY frame for the links in self.ee_idx
    def _rot_per_foot(self, v_w, inv_base_quat):
        # v_w: (N,L,3), inv_base_quat: (N,4)
        N, L, _ = v_w.shape
        v_flat = v_w.reshape(N * L, 3)
        q_flat = inv_base_quat.repeat_interleave(L, dim=0)  # (N·L, 4)
        v_b_flat = transform_by_quat(v_flat, q_flat)  # (N·L, 3)
        return v_b_flat.view(N, L, 3)

    def _feet_vel_body(self):
        links_vel_w = self.robot.get_links_vel()  # (N, n_links, 3)
        base_vel_w = self.robot.get_vel()  # (N, 3)
        v_foot_w = links_vel_w[:, self.ee_idx]  # (N, L, 3)
        v_rel_w = v_foot_w - base_vel_w.unsqueeze(1)  # (N, L, 3)
        inv = inv_quat(self.base_quat)  # (N, 4)
        v_rel_b = self._rot_per_foot(v_rel_w, inv)  # (N, L, 3)
        return v_rel_b

    # Returns (N, L, 3) foot net external contact forces in WORLD frame, plus norms (N, L)
    def _feet_contact_forces(self):
        links_F_w = self.robot.get_links_net_contact_force()  # (N, n_links, 3)
        F_foot_w = links_F_w[:,self.ee_idx]  # (N, L, 3)
        F_norm = torch.linalg.norm(F_foot_w, dim=-1)  # (N, L)
        return F_foot_w, F_norm

    def _reward_object_locomotion(self):
        """
        Drive the base so the ball sits at (radius=r, angle=ψ) in BODY frame, with
        a shaping on nominal speed v_nom toward the target anchor.

        commands: [0]=r [m], [1]=ψ [rad], [2]=v_nom [m/s]
        Uses: self.relative_ball_pos (BODY), self.base_lin_vel (BODY), self.base_ang_vel (BODY)
        """
        r_cmd = self.commands[:, 0].clamp(0.0, 5.0)  # safety clamp
        psi = self.commands[:, 1]
        v_nom = self.commands[:, 2].clamp(0.0, 3.0)

        # target ball position in BODY frame so that at goal the ball is at (r*cosψ, r*sinψ)
        b_xy = self.relative_ball_pos[:, :2]  # (N,2) ball in BODY
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        target_b = torch.stack([r_cmd * cpsi, r_cmd * spsi], dim=1)  # (N,2)

        # position error toward target anchor
        e = (target_b - b_xy)
        d = torch.linalg.norm(e, dim=1) + 1e-6
        u_dir = e / d.unsqueeze(-1)  # unit to target

        # base velocity alignment + speed shaping
        v_xy = self.base_lin_vel[:, :2]
        v_mag = torch.linalg.norm(v_xy, dim=1) + 1e-6
        cos_al = (v_xy * u_dir).sum(dim=1) / v_mag  # ∈[-1,1]
        cos_score = 0.5 * (1.0 + cos_al)  # ∈[0,1]

        d0 = self.reward_cfg.get("obj_loco_d0", 0.6)  # distance scale to fade speed
        v_tgt = v_nom * torch.clamp(d / d0, 0.0, 1.0)
        sigma_v = self.reward_cfg.get("obj_loco_sigma_v", 0.4)
        r_speed = torch.exp(- (v_mag - v_tgt) ** 2 / (2 * sigma_v ** 2))

        sigma_r = self.reward_cfg.get("obj_loco_sigma_r", 0.25)
        r_pos = torch.exp(- (d ** 2) / (2 * sigma_r ** 2))

        # yaw shaping from angular error between current ball bearing and ψ
        ang_b = torch.atan2(b_xy[:, 1], b_xy[:, 0])  # where the ball is now, in BODY
        ang_t = psi  # where we want it
        ang_err = torch.atan2(torch.sin(ang_t - ang_b), torch.cos(ang_t - ang_b))  # wrap to [-π,π]
        k_w = self.reward_cfg.get("obj_loco_k_w", 1.5)
        sigma_w = self.reward_cfg.get("obj_loco_sigma_w", 1.0)
        r_yaw = torch.exp(- (self.base_ang_vel[:, 2] - k_w * ang_err) ** 2 / (2 * sigma_w ** 2))

        # posture/height gate (same spirit as your limb-vel reward)
        upr = (1.0 - self.projected_gravity[:, 2]).clamp(0.0, 2.0) * 0.5
        z = self.base_pos[:, 2]
        z0, z1 = 0.24, 0.30
        gate_h = ((z - z0) / (z1 - z0)).clamp(0.0, 1.0)
        gate = (upr ** 2) * gate_h

        w_pos = self.reward_cfg.get("obj_loco_w_pos", 0.5)
        w_spd = self.reward_cfg.get("obj_loco_w_spd", 0.35)
        w_yaw = self.reward_cfg.get("obj_loco_w_yaw", 0.15)

        return gate * (w_pos * r_pos + w_spd * (cos_score * r_speed) + w_yaw * r_yaw)

    def _reward_tracking_body_pose(self):
        """
        Track (dz, pitch, roll) via height + IMU gravity (yaw-invariant).
        commands: [vx, vy, yaw, dz, pitch, roll]
        """
        # --- targets ---
        dz_t = self.commands[:, 3]
        pitch_t = self.commands[:, 4]
        roll_t = self.commands[:, 5]

        # --- current orientation from gravity in body frame ---
        g = self.projected_gravity  # (N,3), unit, z ~ -1 when upright
        pitch = torch.atan2(g[:, 0], -g[:, 2])
        roll = torch.atan2(-g[:, 1], -g[:, 2])

        # wrap-safe errors
        e_pitch = torch.atan2(torch.sin(pitch - pitch_t), torch.cos(pitch - pitch_t))
        e_roll = torch.atan2(torch.sin(roll - roll_t), torch.cos(roll - roll_t))

        # orientation reward (yaw-free)
        sigma_pitch = self.reward_cfg.get("sigma_pitch", 0.2)  # tune
        sigma_roll = self.reward_cfg.get("sigma_roll", 0.2)
        r_orient = torch.exp(-(e_pitch ** 2) / sigma_pitch - (e_roll ** 2) / sigma_roll)

        # height reward from dz (target height = nominal + dz_t)
        h_nominal = self.reward_cfg.get("base_height_nominal", 0.32)  # your standing height
        h_target = h_nominal + dz_t
        sigma_h = self.reward_cfg.get("sigma_h", 0.01)
        r_height = torch.exp(-torch.square(self.base_pos[:, 2] - h_target) / sigma_h)

        w_o = self.reward_cfg.get("w_pose_orient", 0.5)
        w_h = self.reward_cfg.get("w_pose_height", 0.5)
        return w_o * r_orient + w_h * r_height

    def _reward_tracking_limb_vel(self):
        # --- command & measures ---
        v_cmd = torch.clamp(self.commands[:, 6:9], -1.5, 1.5)  # (N,3)
        s_cmd = torch.linalg.norm(v_cmd, dim=1, keepdim=True)  # (N,1)
        u_cmd = v_cmd / (s_cmd + 1e-6)

        v_foot = self._feet_vel_body()  # (N,L,3)
        v_mag = torch.linalg.norm(v_foot, dim=-1)  # (N,L)

        # per-foot direction + speed
        dot = (v_foot @ u_cmd.unsqueeze(-1)).squeeze(-1)  # (N,L)
        dir_cos = 0.5 * (1.0 + dot / (v_mag + 1e-6))  # [0,1]
        e_speed = (v_mag - s_cmd).squeeze(-1) ** 2  # (N,L)
        sigma_s = self.reward_cfg.get("limb_speed_sigma", 0.45)
        r_speed = torch.exp(- e_speed / (2 * sigma_s ** 2))  # (N,L)
        r_per = torch.sqrt(dir_cos * r_speed + 1e-8)  # (N,L)

        # soft selector + one-hot bonus
        beta = self.reward_cfg.get("limb_selector_beta", 8.0)
        w = torch.softmax(beta * r_per, dim=1)  # (N,L)
        r_track = (w * r_per).sum(dim=1)  # (N,)
        L = r_per.shape[1]
        onehot = (w ** 2).sum(dim=1)
        onehot = (onehot - 1.0 / L) / (1.0 - 1.0 / L)

        # discourage non-selected motion (light)
        p_non = ((1.0 - w) * v_mag).sum(dim=1) * s_cmd.squeeze(-1)

        # --- posture/height gate (no explicit contacts) ---
        # upright scalar in [0,1]: 1 when upright, ~0 when flat
        upr = (1.0 - self.projected_gravity[:, 2]).clamp(0.0, 2.0) * 0.5  # matches _reward_upright
        # smooth step on height
        z = self.base_pos[:, 2]
        z0, z1 = 0.24, 0.30  # no reward if below z0
        gate_h = ((z - z0) / (z1 - z0)).clamp(0.0, 1.0)
        gate = (upr ** 2) * gate_h  # both must be decent

        # final
        return gate * (0.85 * r_track + 0.15 * onehot) - 0.03 * p_non

    def _reward_tracking_contact_hold(self):
        """
        Track normal force and tangential creep for the best contact foot.
        commands[10]=Fn [N], [11]=vt [m/s], [12]=phi [rad].
        """
        Fn_cmd = self.commands[:, 10].clamp(0.0, 500.0)
        vt_cmd = self.commands[:, 11].clamp(-0.5, 0.5)
        phi_cmd = self.commands[:, 12]

        # Forces (WORLD) and feet velocities (BODY)
        F_w, Fn = self._feet_contact_forces()  # (N,L,3), (N,L)
        v_foot_b = self._feet_vel_body()  # (N,L,3)

        # Contact mask from instantaneous force (flags are updated after rewards)
        contact_thresh = self.reward_cfg.get("contact_thresh", 5.0)
        contact = (Fn > contact_thresh).float()  # (N,L)

        eps = 1e-6
        # Unit contact normal (WORLD) → rotate to BODY per foot safely
        n_hat_w = F_w / (Fn.unsqueeze(-1) + eps)  # (N,L,3)
        inv = inv_quat(self.base_quat)  # (N,4)
        n_hat_b = self._rot_per_foot(n_hat_w, inv)  # (N,L,3)

        # Tangent basis in BODY (use ex projected onto plane; fallback to ey if degenerate)
        N, L = Fn.shape
        ex = torch.tensor([1., 0., 0.], device=gs.device, dtype=gs.tc_float).view(1, 1, 3).expand(N, L, 3)
        proj = (ex * n_hat_b).sum(-1, keepdim=True) * n_hat_b
        t1_b = ex - proj
        bad = (torch.linalg.norm(t1_b, dim=-1, keepdim=True) < 1e-3).float()
        ey = torch.tensor([0., 1., 0.], device=gs.device, dtype=gs.tc_float).view(1, 1, 3).expand(N, L, 3)
        t1_fallback = ey - (ey * n_hat_b).sum(-1, keepdim=True) * n_hat_b
        t1_b = torch.where(bad > 0, t1_fallback, t1_b)
        t1_b = t1_b / (torch.linalg.norm(t1_b, dim=-1, keepdim=True) + eps)
        t2_b = torch.cross(n_hat_b, t1_b, dim=-1)

        # Desired BODY tangential direction from phi
        cosφ = torch.cos(phi_cmd).view(-1, 1, 1)  # (N,1,1)
        sinφ = torch.sin(phi_cmd).view(-1, 1, 1)
        t_cmd_b = cosφ * t1_b + sinφ * t2_b  # (N,L,3)

        # Tangential speed along commanded direction
        vt_meas = (v_foot_b * t_cmd_b).sum(dim=-1)  # (N,L)

        # Rewards (masked by contact)
        sigma_F = 20.0
        rF = torch.exp(- (Fn - Fn_cmd.view(-1, 1)).abs() / sigma_F) * contact

        sigma_v = 0.08
        rv = torch.exp(- ((vt_meas - vt_cmd.view(-1, 1)) ** 2) / (2 * sigma_v ** 2)) * contact

        r_per = (rF * rv).sqrt()  # (N,L)
        has_contact = contact.any(dim=1)
        r = torch.where(has_contact, r_per.max(dim=1).values, torch.zeros_like(Fn_cmd))

        # (Optional) light creep penalty on other contact feet
        v_tan = v_foot_b - (v_foot_b * n_hat_b).sum(-1, keepdim=True) * n_hat_b
        v_tan_mag = torch.linalg.norm(v_tan, dim=-1)  # (N,L)
        best_mask = (r_per == r_per.max(dim=1, keepdim=True).values).float()
        creep_pen = ((1.0 - best_mask) * v_tan_mag * contact).sum(dim=1) / (contact.sum(dim=1) + 1e-6)

        return r - 0.05 * creep_pen

    def _reward_tracking_hop(self):
        """
        commands[13]=apex_h [m], [14]=psi [rad] (horizontal takeoff direction in body frame),
        [15]=pitch_imp (scaled target pitch rate).
        Reward terms:
          (1) vertical takeoff speed toward height target
          (2) flight/airtime bonus (all feet off contact)
          (3) horizontal direction alignment (robust at low speed)
          (4) pitch-rate shaping
          (5) (optional) apex-height shaping during flight
        """
        g = 9.81
        h_cmd = self.commands[:, 13].clamp(0.0, 0.40)  # (N,)
        psi = self.commands[:, 14]
        imp = self.commands[:, 15].clamp(-1.0, 1.0)

        # target vertical speed for the commanded apex (v0 = sqrt(2 g h))
        v0 = torch.sqrt(torch.clamp(2.0 * g * h_cmd, min=0.0))  # (N,)
        vz = self.base_lin_vel[:, 2]

        # (1) vertical speed tracking (looser sigma to allow learning)
        sigma_vz = 0.5
        r_vz = torch.exp(- (vz - v0) ** 2 / (2 * sigma_vz ** 2))

        # Foot contacts & flight detection
        F = self.robot.get_links_net_contact_force()  # (N, n_links, 3)
        Fn = torch.linalg.norm(F[:, self.ee_idx], dim=-1)  # (N, 4)
        contact = (Fn > self.reward_cfg.get("contact_thresh", 5.0)).float()  # (N,4)
        num_contact = contact.sum(dim=1)  # (N,)
        in_flight = (num_contact == 0).float()  # (N,)

        # (2) flight/airtime bonus to break the "stand still" basin
        r_flight = in_flight

        # (3) horizontal direction alignment (robust at |v|~0)
        vxy = self.base_lin_vel[:, :2]  # (N,2)
        d = torch.stack([torch.cos(psi), torch.sin(psi)], dim=1)  # (N,2)
        speed = torch.linalg.norm(vxy, dim=1)  # (N,)
        cosang = (vxy * d).sum(dim=1) / (speed + 1e-6)  # (N,)
        r_dir = 0.5 * (1.0 + cosang)  # ∈[0,1], well-defined at low speed

        # (4) pitch-rate shaping (BODY y-axis)
        k_rate = 3.0
        pitch_rate = self.base_ang_vel[:, 1]
        sigma_pitch = 0.8
        r_pitch = torch.exp(- (pitch_rate - k_rate * imp) ** 2 / (2 * sigma_pitch ** 2))

        # (5) apex-height shaping only while in flight
        h0 = self.reward_cfg.get("base_height_target", 0.30)
        h_gain = (self.base_pos[:, 2] - h0).clamp(min=0.0)
        sigma_h = 0.06
        r_apex = torch.exp(- (h_gain - h_cmd) ** 2 / (2 * sigma_h ** 2)) * in_flight

        # Weighted sum (keep simple; avoid geometric mean collapse)
        w_vz, w_f, w_dir, w_pitch, w_h = 0.5, 0.3, 0.1, 0.05, 0.05
        r = w_vz * r_vz + w_f * r_flight + w_dir * r_dir + w_pitch * r_pitch + w_h * r_apex
        return r

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_upright(self):
        """
        Uprightness via gravity projection in body frame.
        projected_gravity[:,2] ≈ -1 when perfectly upright; +1 if upside-down.
        Map to [0,1]: r = 0.5 * (1 - g_z).
        """
        g_z = self.projected_gravity[:, 2].clamp(-1.0, 1.0)
        return 0.5 * (1.0 - g_z)


    '''REWARD FOR ICM'''
    def _reward_survive(self):
        """
        Gives a constant positive reward for every step the agent survives.
        """
        return torch.ones_like(self.rew_buf)

    def _reward_termination(self):
        """
        Applies a large negative penalty for terminating due to reasons other than timeout
        (e.g., falling over).
        """
        # The reset_buf is True for any termination.
        # The extras["time_outs"] is True only for timeout terminations.
        # We penalize terminations that are NOT timeouts.
        failures = torch.logical_and(self.reset_buf, ~self.extras["time_outs"].bool())
        return -failures.float()
