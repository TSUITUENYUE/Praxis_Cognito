import pickle
import genesis as gs
import numpy as np
import time
import torch
import h5py
import os
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from .go2_env import Go2Env
from Model.agent import Agent
import faiss

class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, device='cuda'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

class ICMModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ICMModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, action, next_state):
        state_action = torch.cat([state, action], dim=-1)
        phi = self.encoder(state_action)
        with torch.no_grad():
            phi_next = self.encoder(torch.cat([next_state, action], dim=-1))
        pred_phi_next = self.forward_model(torch.cat([phi, action], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=-1)
        phi_next_with_grad = self.encoder(torch.cat([next_state, action], dim=-1))
        pred_action = self.inverse_model(torch.cat([phi, phi_next_with_grad], dim=-1))
        inverse_loss = F.mse_loss(pred_action, action, reduction='none').mean(dim=-1)
        return forward_loss, inverse_loss

def generate(cfg: DictConfig):
    # --- Config ---
    NUM_ENVS = cfg.dataset.num_envs
    EPISODES_TO_COLLECT = cfg.dataset.episodes
    MAX_EPISODE_SECONDS = cfg.dataset.max_episode_seconds
    FRAME_RATE = cfg.dataset.frame_rate
    AGENT = cfg.agent.name
    path = f"./Pretrain/data/{AGENT}/{NUM_ENVS} {EPISODES_TO_COLLECT} {MAX_EPISODE_SECONDS} {FRAME_RATE}"
    os.makedirs(path, exist_ok=True)
    SAVE_FILENAME = f"{path}/{NUM_ENVS} {EPISODES_TO_COLLECT} {MAX_EPISODE_SECONDS} {FRAME_RATE}.h5"

    CURIOSITY_BETA = cfg.dataset.curiosity_beta
    POLICY_HIDDEN_DIM = cfg.dataset.policy_hidden_dim
    SIMILARITY_THRESHOLD = cfg.dataset.similarity_threshold
    DOWNSAMPLE_FACTOR = 5
    WRITE_BUFFER_SIZE = 5000

    log_dir = cfg.dataset.log_dir
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    original_entropy = train_cfg['algorithm'].get('entropy_coef', 0.0)
    train_cfg['algorithm']['entropy_coef'] = 0.02
    print(f"Updated entropy coefficient from {original_entropy} to {train_cfg['algorithm']['entropy_coef']} to encourage exploration.")

    gs.init(logging_level='warning')

    reward_cfg["reward_scales"] = {"survive": 1.0, "termination": -200.0}
    agent = Agent(**cfg.agent)
    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=NUM_ENVS < 128,
        agent=agent
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    policy_alg = runner.alg

    # ICM (curiosity) — separate normalization
    icm = ICMModule(state_dim=env.num_obs, action_dim=env.num_actions, hidden_dim=POLICY_HIDDEN_DIM).to(gs.device)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-4)

    # --- RMS for ICM (unchanged) ---
    state_dim = env.num_obs
    action_dim = env.num_actions
    state_normalizer = RunningMeanStd((state_dim,), device=gs.device)
    action_normalizer = RunningMeanStd((action_dim,), device=gs.device)
    reward_normalizer = RunningMeanStd((1,), device=gs.device)

    # ★ NEW: RMS for dedup features (global running stats)
    n_dofs = len(env.motors_dof_idx)
    dedup_q_rms   = RunningMeanStd((n_dofs,), device=gs.device)         # joints
    dedup_dq_rms  = RunningMeanStd((n_dofs,), device=gs.device)         # joint velocities
    dedup_ball_rms    = RunningMeanStd((3,), device=gs.device)          # ball pos
    dedup_ballv_rms   = RunningMeanStd((3,), device=gs.device)          # ball vel

    input_dim = env.num_obs
    output_dim = env.num_actions
    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)

    # --- Buffers (GPU) ---
    agent_traj_buffer   = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)
    obj_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, 3),           device=gs.device)
    obs_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, env.num_obs), device=gs.device)
    act_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, env.num_actions), device=gs.device)
    dof_vel_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)
    # ★ NEW: track dones so we can mask after first reset per env
    done_traj_buffer    = torch.zeros((max_episode_len, NUM_ENVS), dtype=torch.bool, device=gs.device)

    embedding_dim = (max_episode_len // DOWNSAMPLE_FACTOR) * (
        n_dofs + n_dofs + 3 + 3   # q, dq, ball pos, ball vel
    )
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)

    total_episodes_saved = 0
    mean_surrogate_loss = 0
    icm_loss = torch.tensor(0.0)
    agent_save_buffer, obj_save_buffer = [], []
    obs_save_buffer, act_save_buffer, vel_save_buffer = [], [], []
    valid_len_save_buffer = []  # ★ NEW

    print(f"Starting curiosity-driven data generation. Target: {EPISODES_TO_COLLECT} episodes.")

    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()

    # Reset with extras; normalize like OnPolicyRunner
    obs, extras = env.reset()
    critic_obs = extras['observations'].get("critic", obs)
    obs_n        = runner.obs_normalizer(obs)
    critic_obs_n = runner.critic_obs_normalizer(critic_obs)


    # Save normalizers for downstream reuse
    norm_state_path = os.path.join(path, "normalizers.pt")
    torch.save({
        "obs_norm": runner.obs_normalizer.state_dict(),
        "critic_obs_norm": runner.critic_obs_normalizer.state_dict(),
    }, norm_state_path)
    print(f"Saved normalizer state to {norm_state_path}")

    with h5py.File(SAVE_FILENAME, 'w') as f:
        agent_ds = f.create_dataset('agent_trajectories',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                    dtype='float32', compression="gzip")
        obj_ds   = f.create_dataset('obj_trajectories',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                    dtype='float32', compression="gzip")
        obs_ds   = f.create_dataset('obs',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, env.num_obs),
                                    dtype='float32', compression="gzip")
        actions_ds = f.create_dataset('actions',
                                      shape=(EPISODES_TO_COLLECT, max_episode_len, env.num_actions),
                                      dtype='float32', compression="gzip")
        dof_vel_ds = f.create_dataset('dof_vel',
                                      shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                      dtype='float32', compression="gzip")
        # ★ NEW: store per-clip valid lengths
        valid_len_ds = f.create_dataset('valid_length',
                                        shape=(EPISODES_TO_COLLECT,), dtype='int32', compression="gzip")

        # Metadata
        meta = f.create_group('meta')
        meta.attrs['dt'] = float(env.dt)
        act_scale = env.env_cfg.get('action_scale', 0.25)
        if isinstance(act_scale, (list, tuple, np.ndarray, torch.Tensor)):
            meta.create_dataset('action_scale', data=np.asarray(act_scale, dtype=np.float32))
        else:
            meta.attrs['action_scale'] = float(act_scale)
        meta.create_dataset('default_dof_pos', data=np.asarray(env.default_dof_pos.cpu(), dtype=np.float32))
        meta.attrs['normalizer_pt'] = os.path.abspath(norm_state_path)
        meta.attrs['faiss_threshold'] = float(SIMILARITY_THRESHOLD)  # ★ NEW
        meta.attrs['downsample_factor'] = int(DOWNSAMPLE_FACTOR)     # ★ NEW

        episodes_since_last_save = 0
        while total_episodes_saved < EPISODES_TO_COLLECT:
            f_loss = torch.tensor(0.0, device=gs.device)
            i_loss = torch.tensor(0.0, device=gs.device)
            action_mean = 0.0
            action_std = 0.0

            # A) Collect a full fixed-length clip per env
            first_done_seen = torch.full((NUM_ENVS,), fill_value=max_episode_len, dtype=torch.int32, device=gs.device)  # ★ NEW
            for t in range(max_episode_len):
                # log obs_t (store raw)
                obs_traj_buffer[t] = obs

                # act with *normalized* obs like OnPolicyRunner
                actions = policy_alg.act(obs_n, critic_obs_n)
                act_traj_buffer[t] = actions

                # step env to t+1
                next_obs, rews, dones, infos = env.step(actions)
                done_traj_buffer[t] = (dones.squeeze(-1) > 0) if dones.ndim == 2 else (dones > 0)

                # update dedup RMS from kinematics and ball (global stats) ★ NEW
                q_now  = env.robot.get_dofs_position(env.motors_dof_idx)
                dq_now = env.robot.get_dofs_velocity(env.motors_dof_idx)
                ball_now = env.ball.get_pos()
                dedup_q_rms.update(q_now)
                dedup_dq_rms.update(dq_now)
                dedup_ball_rms.update(ball_now)

                # curiosity plumbing (ICM uses its own RMS)
                state_normalizer.update(obs)
                action_normalizer.update(actions)
                state_normalizer.update(next_obs)
                norm_obs, norm_actions, norm_next_obs = (
                    state_normalizer.normalize(obs),
                    action_normalizer.normalize(actions),
                    state_normalizer.normalize(next_obs),
                )
                with torch.no_grad():
                    intrinsic_reward, _ = icm(norm_obs, norm_actions, norm_next_obs)
                reward_normalizer.update(intrinsic_reward.unsqueeze(-1))
                intrinsic_reward /= torch.sqrt(reward_normalizer.var + reward_normalizer.epsilon)
                total_reward = rews + intrinsic_reward * CURIOSITY_BETA

                # PPO storage update
                policy_alg.process_env_step(total_reward, dones, infos)

                # record kinematics AFTER step (post a_t)
                agent_traj_buffer[t]   = q_now
                dof_vel_traj_buffer[t] = dq_now
                obj_traj_buffer[t]     = ball_now

                # advance raw obs + compute normalized obs for next action using runner's normalizers
                obs = next_obs
                if "observations" in infos and "critic" in infos["observations"]:
                    critic_obs = infos["observations"]["critic"]
                else:
                    critic_obs = obs
                obs_n        = runner.obs_normalizer(obs)
                critic_obs_n = runner.critic_obs_normalizer(critic_obs)

                # record first done index per env (to mask later) ★ NEW
                freshly_done = (done_traj_buffer[t] & (first_done_seen == max_episode_len))
                if freshly_done.any():
                    idxs = torch.nonzero(freshly_done, as_tuple=False).squeeze(-1)
                    first_done_seen[idxs] = t + 1  # valid length is up to and including this step

                # PPO update cadence
                if policy_alg.storage.step >= runner.num_steps_per_env:
                    num_transitions = policy_alg.storage.step
                    if num_transitions > 1:
                        all_states  = policy_alg.storage.observations.view(-1, input_dim)
                        all_actions = policy_alg.storage.actions.view(-1, output_dim)
                        end_idx = (num_transitions - 1) * policy_alg.storage.num_envs
                        batch_states, batch_actions = all_states[:end_idx], all_actions[:end_idx]
                        start_idx_next, end_idx_next = policy_alg.storage.num_envs, num_transitions * policy_alg.storage.num_envs
                        batch_next_states = all_states[start_idx_next:end_idx_next]

                        norm_batch_states      = state_normalizer.normalize(batch_states)
                        norm_batch_actions     = action_normalizer.normalize(batch_actions)
                        norm_batch_next_states = state_normalizer.normalize(batch_next_states)

                        f_loss, i_loss = icm(norm_batch_states, norm_batch_actions, norm_batch_next_states)
                        icm_loss = f_loss.mean() + i_loss.mean()
                        icm_optimizer.zero_grad()
                        icm_loss.backward()
                        icm_optimizer.step()

                    policy_alg.compute_returns(critic_obs_n)  # ★ CHANGED: use critic obs (normalized)
                    mean_value_loss, mean_surrogate_loss, _, _, _ = policy_alg.update()

            action_mean = torch.mean(actions).item()
            action_std  = torch.std(actions).item()

            # B) Move to CPU + build dedup embeddings
            agent_np = agent_traj_buffer.cpu().numpy()    # [T, E, DoF]
            obj_np   = obj_traj_buffer.cpu().numpy()      # [T, E, 3]
            obs_np   = obs_traj_buffer.cpu().numpy()
            act_np   = act_traj_buffer.cpu().numpy()
            vel_np   = dof_vel_traj_buffer.cpu().numpy()  # [T, E, DoF]
            done_np  = done_traj_buffer.cpu().numpy()     # [T, E]
            # valid length per env clip = first done if seen, else max len
            valid_len_np = np.where(first_done_seen.cpu().numpy() == max_episode_len,
                                    max_episode_len,
                                    first_done_seen.cpu().numpy())

            # transpose to [E, T, ...]
            agent_batch = np.transpose(agent_np, (1, 0, 2))
            obj_batch   = np.transpose(obj_np,   (1, 0, 2))
            obs_batch   = np.transpose(obs_np,   (1, 0, 2))
            act_batch   = np.transpose(act_np,   (1, 0, 2))
            vel_batch   = np.transpose(vel_np,   (1, 0, 2))
            done_batch  = np.transpose(done_np,  (1, 0))    # [E, T]

            # ★ NEW: mask frames after first reset per env to avoid mixing clips
            for e in range(NUM_ENVS):
                L = valid_len_np[e]
                if L < max_episode_len:
                    agent_batch[e, L:] = 0.0
                    vel_batch[e,   L:] = 0.0
                    obj_batch[e,   L:] = 0.0
                    obs_batch[e,   L:] = 0.0
                    act_batch[e,   L:] = 0.0

            # ★ NEW: compute ball velocity (central difference with padding)
            ball_vel_batch = np.zeros_like(obj_batch)
            ball_vel_batch[:, 1:-1, :] = 0.5 * (obj_batch[:, 2:, :] - obj_batch[:, :-2, :])
            ball_vel_batch[:, 0,   :] = (obj_batch[:, 1,  :] - obj_batch[:, 0,  :])
            ball_vel_batch[:, -1,  :] = (obj_batch[:, -1, :] - obj_batch[:, -2, :])

            # Downsample all streams on time axis
            agent_dsmp   = agent_batch[:, ::DOWNSAMPLE_FACTOR, :]     # [E, Td, DoF]
            vel_dsmp     = vel_batch[:,   ::DOWNSAMPLE_FACTOR, :]
            ball_dsmp    = obj_batch[:,   ::DOWNSAMPLE_FACTOR, :]     # [E, Td, 3]
            ballv_dsmp   = ball_vel_batch[:, ::DOWNSAMPLE_FACTOR, :]

            # ★ NEW: z-score per feature using global RMS from GPU
            q_mean  = dedup_q_rms.mean.cpu().numpy()
            q_std   = np.sqrt(dedup_q_rms.var.cpu().numpy() + 1e-6)
            dq_mean = dedup_dq_rms.mean.cpu().numpy()
            dq_std  = np.sqrt(dedup_dq_rms.var.cpu().numpy() + 1e-6)
            b_mean  = dedup_ball_rms.mean.cpu().numpy()
            b_std   = np.sqrt(dedup_ball_rms.var.cpu().numpy() + 1e-6)
            bv_mean = dedup_ballv_rms.mean.cpu().numpy()
            bv_std  = np.sqrt(dedup_ballv_rms.var.cpu().numpy() + 1e-6)

            agent_z = (agent_dsmp - q_mean) / q_std
            vel_z   = (vel_dsmp   - dq_mean) / dq_std
            ball_z  = (ball_dsmp  - b_mean) / b_std
            ballv_z = (ballv_dsmp - bv_mean) / bv_std

            # Flatten to embeddings and L2-normalize → cosine similarity under IndexFlatIP
            emb_parts = [
                agent_z.reshape(NUM_ENVS, -1),
                vel_z.reshape(NUM_ENVS,   -1),
                ball_z.reshape(NUM_ENVS,  -1),
                ballv_z.reshape(NUM_ENVS, -1)
            ]
            new_embeddings = np.concatenate(emb_parts, axis=1).astype(np.float32)
            faiss.normalize_L2(new_embeddings)

            # FAISS dedup: keep if max cosine similarity < τ
            if faiss_index.ntotal == 0:
                unique_indices = np.arange(len(new_embeddings))
            else:
                D, I = faiss_index.search(x=new_embeddings, k=1)  # D: cosine similarity in [-1,1]
                is_unique_mask = D[:, 0] < SIMILARITY_THRESHOLD
                unique_indices = np.where(is_unique_mask)[0]

            if unique_indices.size > 0:
                agent_save_buffer.append(agent_batch[unique_indices])
                obj_save_buffer.append(obj_batch[unique_indices])
                obs_save_buffer.append(obs_batch[unique_indices])
                act_save_buffer.append(act_batch[unique_indices])
                vel_save_buffer.append(vel_batch[unique_indices])
                valid_len_save_buffer.append(valid_len_np[unique_indices])  # ★ NEW
                faiss_index.add(x=new_embeddings[unique_indices])
                episodes_since_last_save += len(unique_indices)

            # Write in blocks
            if episodes_since_last_save >= WRITE_BUFFER_SIZE or total_episodes_saved + episodes_since_last_save >= EPISODES_TO_COLLECT:
                if agent_save_buffer:
                    agent_block = np.concatenate(agent_save_buffer, axis=0)
                    obj_block   = np.concatenate(obj_save_buffer,   axis=0)
                    obs_block   = np.concatenate(obs_save_buffer,   axis=0)
                    act_block   = np.concatenate(act_save_buffer,   axis=0)
                    vel_block   = np.concatenate(vel_save_buffer,   axis=0)
                    vlen_block  = np.concatenate(valid_len_save_buffer, axis=0)

                    num_in_block = agent_block.shape[0]
                    start_idx = total_episodes_saved
                    end_idx = min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
                    num_to_write = end_idx - start_idx
                    if num_to_write > 0:
                        print(f"  ...Writing {num_to_write} episodes to disk...")
                        agent_ds[start_idx:end_idx]   = agent_block[:num_to_write]
                        obj_ds[start_idx:end_idx]     = obj_block[:num_to_write]
                        obs_ds[start_idx:end_idx]     = obs_block[:num_to_write]
                        actions_ds[start_idx:end_idx] = act_block[:num_to_write]
                        dof_vel_ds[start_idx:end_idx] = vel_block[:num_to_write]
                        valid_len_ds[start_idx:end_idx] = vlen_block[:num_to_write]  # ★ NEW
                        total_episodes_saved = end_idx

                    agent_save_buffer, obj_save_buffer = [], []
                    obs_save_buffer, act_save_buffer, vel_save_buffer = [], [], []
                    valid_len_save_buffer = []  # ★ NEW
                    episodes_since_last_save = 0

            print(f"  ...Collected: {total_episodes_saved}/{EPISODES_TO_COLLECT} | PPO Loss: {mean_surrogate_loss:.3f} "
                  f"| Fwd Loss: {f_loss.mean().item():.3f} | Inv Loss: {i_loss.mean().item():.3f} "
                  f"| Action Mean: {action_mean:.3f} | Action Std: {action_std:.3f}")

        # Final flush if remaining
        if agent_save_buffer:
            agent_block = np.concatenate(agent_save_buffer, axis=0)
            obj_block   = np.concatenate(obj_save_buffer,   axis=0)
            obs_block   = np.concatenate(obs_save_buffer,   axis=0)
            act_block   = np.concatenate(act_save_buffer,   axis=0)
            vel_block   = np.concatenate(vel_save_buffer,   axis=0)
            vlen_block  = np.concatenate(valid_len_save_buffer, axis=0)

            num_in_block = agent_block.shape[0]
            start_idx = total_episodes_saved
            end_idx = min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
            num_to_write = end_idx - start_idx
            if num_to_write > 0:
                print(f"  ...Writing final {num_to_write} episodes to disk...")
                agent_ds[start_idx:end_idx]   = agent_block[:num_to_write]
                obj_ds[start_idx:end_idx]     = obj_block[:num_to_write]
                obs_ds[start_idx:end_idx]     = obs_block[:num_to_write]
                actions_ds[start_idx:end_idx] = act_block[:num_to_write]
                dof_vel_ds[start_idx:end_idx] = vel_block[:num_to_write]
                valid_len_ds[start_idx:end_idx] = vlen_block[:num_to_write]  # ★ NEW
                total_episodes_saved = end_idx

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Normalizers saved to '{norm_state_path}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")
