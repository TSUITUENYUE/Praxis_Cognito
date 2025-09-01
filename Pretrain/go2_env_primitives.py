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

        self.ee_idx = agent.end_effector
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
        self.commands_scale = 0
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
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

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

        links_F = self.robot.get_links_net_contact_force()

        # pick feet: (N, 4, 3) → norm: (N, 4)
        F_foot = links_F.index_select(dim=1, index=self.ee_idx)
        F_norm = torch.linalg.norm(F_foot, dim=-1)

        # hysteresis: turn on when > ON, turn off when < OFF, otherwise hold previous
        on = (F_norm > self.contact_on_force_N)
        off = (F_norm < self.contact_off_force_N)
        # keep dtype consistent with obs_buf
        self.contact_flags = torch.where(on, 1.0, torch.where(off, 0.0, self.contact_flags))
        # compute observations
        base_height = self.base_pos[:, 2:3]

        self.obs_buf = torch.cat(
            [
                self.commands * self.commands_scale,  # 14
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                base_height,  # 1   <-- NEW
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # A
                self.dof_vel * self.obs_scales["dof_vel"],  # A
                exec_actions,  # A
                self.contact_flags,  # 4   <-- NEW
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
        ball_pos[:, 0] = gs_rand_float(0.5, 1.5, (len(envs_idx),), gs.device)
        ball_pos[:, 1] = gs_rand_float(-0.5, 0.5, (len(envs_idx),), gs.device)
        ball_pos[:, 2] = gs_rand_float(0.0, 0.5, (len(envs_idx),), gs.device)
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
    def _feet_vel_body(self):
        # Adapt to your Genesis build if names differ:
        # Many builds provide get_links_velocity() / get_links_pos()
        links_vel_w = self.robot.get_links_velocity()  # (N, n_links, 3)
        base_vel_w = self.robot.get_vel()  # (N, 3)
        inv_base_quat = inv_quat(self.base_quat)  # (N, 4)
        v_foot_w = links_vel_w.index_select(dim=1, index=self.ee_idx)  # (N, L, 3)
        v_rel_w = v_foot_w - base_vel_w.unsqueeze(1)  # subtract base linear vel
        # transform world→body per env, per foot
        v_rel_b = transform_by_quat(v_rel_w, inv_base_quat.unsqueeze(1))  # (N, L, 3)
        return v_rel_b

    # Returns (N, L, 3) foot net external contact forces in WORLD frame, plus norms (N, L)
    def _feet_contact_forces(self):
        links_F_w = self.robot.get_links_net_contact_force()  # (N, n_links, 3)
        F_foot_w = links_F_w.index_select(dim=1, index=self.ee_idx)  # (N, L, 3)
        F_norm = torch.linalg.norm(F_foot_w, dim=-1)  # (N, L)
        return F_foot_w, F_norm

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_body_pose(self):
        """
        Track body pose deltas: commands[3]=Δz, [4]=Δpitch, [5]=Δroll
        Height target is absolute: base_height_target + Δz.
        """
        dz_cmd = self.commands[:, 3]
        pitch_cmd = self.commands[:, 4]
        roll_cmd = self.commands[:, 5]

        # targets
        h_tgt = self.reward_cfg["base_height_target"] + dz_cmd

        # current
        h = self.base_pos[:, 2]
        # base_euler computed in step(): [roll, pitch, yaw]
        pitch = self.base_euler[:, 1]
        roll = self.base_euler[:, 0]

        # errors
        e_h = h - h_tgt
        e_pitch = self._angle_wrap(pitch - pitch_cmd)
        e_roll = self._angle_wrap(roll - roll_cmd)

        sigma_h, sigma_ang = 0.03, 0.10  # meters, radians (tune if needed)
        r_h = torch.exp(- (e_h ** 2) / (2 * sigma_h ** 2))
        r_p = torch.exp(- (e_pitch ** 2) / (2 * sigma_ang ** 2))
        r_r = torch.exp(- (e_roll ** 2) / (2 * sigma_ang ** 2))

        # geometric mean -> balanced tradeoff
        r = (r_h * r_p * r_r) ** (1 / 3)
        return r

    def _reward_tracking_limb_vel(self):
        """
        Track end-effector Cartesian velocity (body frame).
        commands[6:9] = [vx, vy, vz], commands[9] = impedance scale (not used in reward).
        Rewards the best-matching foot to avoid needing a limb id.
        """
        v_cmd = self.commands[:, 6:9]  # (N, 3)
        v_cmd = torch.clamp(v_cmd, -10.0, 10.0)

        v_foot_b = self._feet_vel_body()  # (N, L, 3)
        # error per foot
        e = v_foot_b - v_cmd.unsqueeze(1)  # (N, L, 3)
        e2 = torch.sum(e * e, dim=-1)  # (N, L)

        sigma_v = 0.25  # m/s tolerance
        r_per_foot = torch.exp(- e2 / (2 * sigma_v ** 2))  # (N, L)
        r = torch.max(r_per_foot, dim=1).values  # best foot wins
        return r

    def _reward_tracking_contact_hold(self):
        """
        Track normal force and tangential creep when in contact.
        commands[10]=Fn [N], [11]=vt [m/s], [12]=phi [rad] (direction in contact plane).
        Rewards the best contact foot.
        """
        Fn_cmd = self.commands[:, 10].clamp(0.0, 500.0)  # (N,)
        vt_cmd = self.commands[:, 11].clamp(-0.5, 0.5)  # (N,)
        phi_cmd = self.commands[:, 12]  # (N,)

        F_w, Fn = self._feet_contact_forces()  # (N,L,3), (N,L)
        v_foot_b = self._feet_vel_body()  # (N,L,3)

        # contact mask from hysteresis flags (N,L)
        contact = (self.contact_flags > 0.5)

        # Build proxy normals (WORLD) and tangent basis in BODY frame
        eps = 1e-6
        n_hat_w = F_w / (Fn.unsqueeze(-1) + eps)  # (N,L,3), stable when in contact
        # pick a body-x world direction by rotating body-x to world
        body_x_b = torch.tensor([1., 0., 0.], device=gs.device, dtype=gs.tc_float).view(1, 1, 3).repeat(self.num_envs,
                                                                                                        1, 1)
        body_x_w = transform_by_quat(body_x_b, self.base_quat.unsqueeze(1))  # (N,1,3)
        t1_w = body_x_w - (n_hat_w * torch.sum(body_x_w * n_hat_w, dim=-1, keepdim=True))
        t1_w = t1_w / (torch.linalg.norm(t1_w, dim=-1, keepdim=True) + eps)
        t2_w = torch.cross(n_hat_w, t1_w, dim=-1)

        # Bring tangents to BODY for projecting body-frame foot velocities
        inv_base_quat = inv_quat(self.base_quat)
        t1_b = transform_by_quat(t1_w, inv_base_quat.unsqueeze(1))
        t2_b = transform_by_quat(t2_w, inv_base_quat.unsqueeze(1))

        # Desired tangential direction in BODY plane
        cosφ = torch.cos(phi_cmd).unsqueeze(1)  # (N,1)
        sinφ = torch.sin(phi_cmd).unsqueeze(1)
        # Combine basis per-foot; broadcast φ per env across feet
        t_cmd_b = cosφ.unsqueeze(-1) * t1_b + sinφ.unsqueeze(-1) * t2_b  # (N,L,3)

        # Project measured body-frame foot velocity onto t_cmd
        vt_meas = torch.sum(v_foot_b * t_cmd_b, dim=-1)  # (N,L)

        # --- rewards ---
        # Force tracking (only when contact)
        sigma_F = 10.0
        rF = torch.exp(- (Fn - Fn_cmd.unsqueeze(1)).abs() / sigma_F) * contact

        # Tangential speed tracking (only when contact)
        sigma_v = 0.05
        rv = torch.exp(- ((vt_meas - vt_cmd.unsqueeze(1)) ** 2) / (2 * sigma_v ** 2)) * contact

        # Combine and pick best contact foot
        r_per_foot = (rF * rv) ** 0.5  # geometric mean
        r = torch.where(contact.any(dim=1),
                        r_per_foot.max(dim=1).values,
                        torch.zeros_like(Fn_cmd))
        return r

    def _reward_tracking_hop(self):
        """
        commands[13]=apex_h [m], [14]=psi [rad] (horizontal takeoff direction in body frame),
        [15]=pitch_imp (scaled target pitch rate).
        Two components:
          - vertical speed toward height target (pre/at takeoff),
          - horizontal velocity direction toward psi,
          - pitch rate toward scaled command.
        """
        h_cmd = self.commands[:, 13].clamp(0.0, 0.40)  # (N,)
        psi_cmd = self.commands[:, 14]
        imp_cmd = self.commands[:, 15].clamp(-1.0, 1.0)

        g = 9.81
        v0 = torch.sqrt(torch.clamp(2.0 * g * torch.abs(h_cmd), min=0.0))  # target vertical speed for hop height
        vz = self.base_lin_vel[:, 2].abs()

        # vertical speed tracking
        sigma_vz = 0.2
        r_vz = torch.exp(- (vz - v0) ** 2 / (2 * sigma_vz ** 2))

        # horizontal direction tracking (in BODY frame)
        vx, vy = self.base_lin_vel[:, 0], self.base_lin_vel[:, 1]
        psi_meas = torch.atan2(vy, vx)
        dpsi = torch.abs(
            torch.tensor([self._angle_wrap(p - q) for p, q in zip(psi_meas, psi_cmd)]).to(gs.device).to(gs.tc_float))
        sigma_psi = 0.35  # rad
        r_dir = torch.exp(- dpsi ** 2 / (2 * sigma_psi ** 2))

        # pitch impulse as pitch rate target (BODY y angular vel)
        # scale desired |pitch rate| by a constant factor per unit imp_cmd
        k_rate = 3.0  # rad/s per unit impulse (tune)
        pitch_rate_meas = self.base_ang_vel[:, 1]
        r_pitch = torch.exp(- (pitch_rate_meas - k_rate * imp_cmd) ** 2 / (2 * (0.6 ** 2)))

        # Combine (weighted geometric mean)
        r = (r_vz * (r_dir ** 0.5) * (r_pitch ** 0.5)) ** (1 / 2)
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
