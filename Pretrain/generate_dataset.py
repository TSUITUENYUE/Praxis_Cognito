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
import logging

class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, device='cuda'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
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
    # Data Gen Configs
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
    icm = ICMModule(state_dim=env.num_obs, action_dim=env.num_actions, hidden_dim=POLICY_HIDDEN_DIM).to(gs.device)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-4)

    state_dim = env.num_obs
    action_dim = env.num_actions
    state_normalizer = RunningMeanStd((state_dim,), device=gs.device)
    action_normalizer = RunningMeanStd((action_dim,), device=gs.device)
    reward_normalizer = RunningMeanStd((1,), device=gs.device)

    n_dofs = len(env.motors_dof_idx)
    input_dim = env.num_obs
    output_dim = env.num_actions
    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)

    # --- Buffers (GPU) ---
    agent_traj_buffer   = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)
    obj_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, 3),           device=gs.device)
    obs_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, env.num_obs), device=gs.device)
    act_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, env.num_actions), device=gs.device)
    dof_vel_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)

    embedding_dim = (max_episode_len // DOWNSAMPLE_FACTOR) * n_dofs
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)

    total_episodes_saved = 0
    mean_surrogate_loss = 0
    icm_loss = torch.tensor(0.0)
    agent_save_buffer, obj_save_buffer = [], []
    obs_save_buffer, act_save_buffer, vel_save_buffer = [], [], []

    print(f"Starting curiosity-driven data generation. Target: {EPISODES_TO_COLLECT} episodes.")

    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()
    obs, _ = env.reset()

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

        # Helpful metadata for training later
        meta = f.create_group('meta')
        meta.attrs['dt'] = float(env.dt)
        meta.attrs['action_scale'] = float(env.env_cfg.get('action_scale', 0.25))
        meta.attrs['default_dof_pos'] = np.asarray(env.default_dof_pos.cpu(), dtype=np.float32)

        episodes_since_last_save = 0
        while total_episodes_saved < EPISODES_TO_COLLECT:
            f_loss = torch.tensor(0.0, device=gs.device)
            i_loss = torch.tensor(0.0, device=gs.device)
            action_mean = 0.0
            action_std = 0.0

            # A) Collect a full episode
            for t in range(max_episode_len):
                # log obs_t, choose action a_t
                obs_traj_buffer[t] = obs
                actions = policy_alg.act(obs, obs)
                act_traj_buffer[t] = actions

                # step env to t+1
                next_obs, rews, dones, infos = env.step(actions)

                # curiosity plumbing
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
                policy_alg.process_env_step(total_reward, dones, infos)

                # record kinematics AFTER step (post a_t)
                agent_traj_buffer[t]   = env.robot.get_dofs_position(env.motors_dof_idx)
                dof_vel_traj_buffer[t] = env.robot.get_dofs_velocity(env.motors_dof_idx)
                obj_traj_buffer[t]     = env.ball.get_pos()

                obs = next_obs

                if policy_alg.storage.step >= runner.num_steps_per_env:
                    num_transitions = policy_alg.storage.step
                    if num_transitions > 1:
                        all_states  = policy_alg.storage.observations.view(-1, input_dim)
                        all_actions = policy_alg.storage.actions.view(-1, output_dim)
                        end_idx = (num_transitions - 1) * policy_alg.storage.num_envs
                        batch_states, batch_actions = all_states[:end_idx], all_actions[:end_idx]
                        start_idx_next, end_idx_next = policy_alg.storage.num_envs, num_transitions * policy_alg.storage.num_envs
                        batch_next_states = all_states[start_idx_next:end_idx_next]

                        norm_batch_states     = state_normalizer.normalize(batch_states)
                        norm_batch_actions    = action_normalizer.normalize(batch_actions)
                        norm_batch_next_states= state_normalizer.normalize(batch_next_states)

                        f_loss, i_loss = icm(norm_batch_states, norm_batch_actions, norm_batch_next_states)
                        icm_loss = f_loss.mean() + i_loss.mean()
                        icm_optimizer.zero_grad()
                        icm_loss.backward()
                        icm_optimizer.step()
                    policy_alg.compute_returns(obs)
                    mean_value_loss, mean_surrogate_loss, _, _, _ = policy_alg.update()

            action_mean = torch.mean(actions).item()
            action_std  = torch.std(actions).item()

            # B) Move to CPU + dedup by joints
            agent_np = agent_traj_buffer.cpu().numpy()
            obj_np   = obj_traj_buffer.cpu().numpy()
            obs_np   = obs_traj_buffer.cpu().numpy()
            act_np   = act_traj_buffer.cpu().numpy()
            vel_np   = dof_vel_traj_buffer.cpu().numpy()

            agent_batch = np.transpose(agent_np, (1, 0, 2))
            obj_batch   = np.transpose(obj_np,   (1, 0, 2))
            obs_batch   = np.transpose(obs_np,   (1, 0, 2))
            act_batch   = np.transpose(act_np,   (1, 0, 2))
            vel_batch   = np.transpose(vel_np,   (1, 0, 2))

            downsampled = agent_batch[:, ::DOWNSAMPLE_FACTOR, :]
            new_embeddings = downsampled.reshape(NUM_ENVS, -1).astype(np.float32)
            faiss.normalize_L2(new_embeddings)

            if faiss_index.ntotal == 0:
                unique_indices = np.arange(len(new_embeddings))
            else:
                D, I = faiss_index.search(x=new_embeddings, k=1)
                is_unique_mask = D[:, 0] < SIMILARITY_THRESHOLD
                unique_indices = np.where(is_unique_mask)[0]

            if unique_indices.size > 0:
                agent_save_buffer.append(agent_batch[unique_indices])
                obj_save_buffer.append(obj_batch[unique_indices])
                obs_save_buffer.append(obs_batch[unique_indices])
                act_save_buffer.append(act_batch[unique_indices])
                vel_save_buffer.append(vel_batch[unique_indices])
                faiss_index.add(x=new_embeddings[unique_indices])
                episodes_since_last_save += len(unique_indices)

            if episodes_since_last_save >= WRITE_BUFFER_SIZE or total_episodes_saved + episodes_since_last_save >= EPISODES_TO_COLLECT:
                if agent_save_buffer:
                    agent_block = np.concatenate(agent_save_buffer, axis=0)
                    obj_block   = np.concatenate(obj_save_buffer,   axis=0)
                    obs_block   = np.concatenate(obs_save_buffer,   axis=0)
                    act_block   = np.concatenate(act_save_buffer,   axis=0)
                    vel_block   = np.concatenate(vel_save_buffer,   axis=0)

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
                        total_episodes_saved = end_idx

                    agent_save_buffer, obj_save_buffer = [], []
                    obs_save_buffer, act_save_buffer, vel_save_buffer = [], [], []
                    episodes_since_last_save = 0

            print(f"  ...Collected: {total_episodes_saved}/{EPISODES_TO_COLLECT} | PPO Loss: {mean_surrogate_loss:.3f} "
                  f"| Fwd Loss: {f_loss.mean().item():.3f} | Inv Loss: {i_loss.mean().item():.3f} "
                  f"| Action Mean: {action_mean:.3f} | Action Std: {action_std:.3f}")

        # Final write for remaining items
        if agent_save_buffer:
            agent_block = np.concatenate(agent_save_buffer, axis=0)
            obj_block   = np.concatenate(obj_save_buffer,   axis=0)
            obs_block   = np.concatenate(obs_save_buffer,   axis=0)
            act_block   = np.concatenate(act_save_buffer,   axis=0)
            vel_block   = np.concatenate(vel_save_buffer,   axis=0)

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
                total_episodes_saved = end_idx

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")
