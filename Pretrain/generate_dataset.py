import pickle
import genesis as gs
import taichi as ti
import numpy as np
import time
import torch
import h5py
import os
import hydra
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from .go2_env import Go2Env

class BatchedRandomPolicyNN(nn.Module):
    def __init__(self, num_envs, input_dim, hidden_dim, output_dim):
        super(BatchedRandomPolicyNN, self).__init__()
        self.num_envs = num_envs

        # Create batched weights and biases
        self.fc1_w = nn.Parameter(torch.empty(num_envs, hidden_dim, input_dim))
        self.fc1_b = nn.Parameter(torch.empty(num_envs, 1, hidden_dim))
        self.fc2_w = nn.Parameter(torch.empty(num_envs, hidden_dim, hidden_dim))
        self.fc2_b = nn.Parameter(torch.empty(num_envs, 1, hidden_dim))
        self.fc3_w = nn.Parameter(torch.empty(num_envs, output_dim, hidden_dim))
        self.fc3_b = nn.Parameter(torch.empty(num_envs, 1, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Vectorized initialization for all policies at once
        for param in self.parameters():
            if param.dim() > 2: # Weights
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            else: # Biases
                nn.init.zeros_(param)

    def forward(self, x):
        # x shape: (num_envs, 1, input_dim)
        # Use batched matrix multiply (bmm)
        x = F.relu(torch.bmm(x, self.fc1_w.transpose(1, 2)) + self.fc1_b)
        x = F.relu(torch.bmm(x, self.fc2_w.transpose(1, 2)) + self.fc2_b)
        x = torch.bmm(x, self.fc3_w.transpose(1, 2)) + self.fc3_b
        return x.squeeze(1) # Output shape: (num_envs, output_dim)


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

    STRUCTURED_FRACTION = cfg.dataset.structured_fraction
    NUM_STRUCTURED = int(EPISODES_TO_COLLECT * STRUCTURED_FRACTION)
    NUM_RANDOM = EPISODES_TO_COLLECT - NUM_STRUCTURED
    NOISE_STD_MAX = cfg.dataset.noise_std_max
    POLICY_HIDDEN_DIM = cfg.dataset.policy_hidden_dim

    log_dir = cfg.dataset.log_dir
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    gs.init()
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=NUM_ENVS < 128,
    )

    # Load RL policy once
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model.pt")
    runner.load(resume_path)
    rl_policy = runner.get_inference_policy(device=gs.device)

    # Joint/DoF from env
    n_dofs = len(env.motors_dof_idx)
    input_dim = env.num_obs
    output_dim = env.num_actions

    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)
    total_episodes_saved = 0

    agent_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs), device=gs.device)
    obj_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, 3), device=gs.device)

    print(f"Starting data generation. Target: {EPISODES_TO_COLLECT} episodes (Structured: {NUM_STRUCTURED}, Random: {NUM_RANDOM}).")
    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()

    with h5py.File(SAVE_FILENAME, 'w') as f:
        agent_ds = f.create_dataset('agent_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs), dtype='float32', compression="gzip")
        obj_ds = f.create_dataset('obj_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, 3), dtype='float32', compression="gzip")

        # --- Structured episodes (RL policy + gradual noise) ---
        structured_saved = 0
        while structured_saved < NUM_STRUCTURED:
            obs, _ = env.reset()
            with torch.no_grad():
                for t in range(max_episode_len):
                    agent_traj_buffer[t] = env.robot.get_dofs_position(env.motors_dof_idx)
                    obj_traj_buffer[t] = torch.zeros((NUM_ENVS, 3), device=gs.device)

                    actions = rl_policy(obs)
                    noise_std = NOISE_STD_MAX * (t / max_episode_len)
                    noise = torch.normal(0, noise_std, size=actions.shape, device=gs.device)
                    actions_noisy = actions + noise
                    obs, rews, dones, infos = env.step(actions_noisy)

            start_idx = total_episodes_saved
            num_this_batch = min(NUM_ENVS, NUM_STRUCTURED - structured_saved)
            end_idx = start_idx + num_this_batch

            # ✅ Slice on GPU first, then copy only the necessary data
            agent_data_to_save = agent_traj_buffer[:, :num_this_batch, :].cpu().numpy()
            obj_data_to_save = obj_traj_buffer[:, :num_this_batch, :].cpu().numpy()

            agent_ds[start_idx:end_idx] = np.transpose(agent_data_to_save, (1, 0, 2))
            obj_ds[start_idx:end_idx] = np.transpose(obj_data_to_save, (1, 0, 2))

            structured_saved += num_this_batch
            total_episodes_saved += num_this_batch
            print(f"  ...Collected structured episodes: {structured_saved}/{NUM_STRUCTURED}")

        # --- Random episodes (random NN policies) ---
        random_saved = 0
        # ✅ Create a single batched policy for all environments
        batched_policy = BatchedRandomPolicyNN(NUM_ENVS, input_dim, POLICY_HIDDEN_DIM, output_dim).to(gs.device)

        while random_saved < NUM_RANDOM:
            obs, _ = env.reset()
            # ✅ Re-initialize all policies at once efficiently
            batched_policy.reset_parameters()

            with torch.no_grad():
                for t in range(max_episode_len):
                    agent_traj_buffer[t] = env.robot.get_dofs_position(env.motors_dof_idx)
                    obj_traj_buffer[t] = torch.zeros((NUM_ENVS, 3), device=gs.device)

                    # ✅ Use a single, fast, batched forward pass
                    actions = batched_policy(obs.unsqueeze(1))
                    obs, rews, dones, infos = env.step(actions)

            start_idx = total_episodes_saved
            num_this_batch = min(NUM_ENVS, NUM_RANDOM - random_saved)
            end_idx = start_idx + num_this_batch

            # ✅ Slice on GPU first, then copy only the necessary data
            agent_data_to_save = agent_traj_buffer[:, :num_this_batch, :].cpu().numpy()
            obj_data_to_save = obj_traj_buffer[:, :num_this_batch, :].cpu().numpy()

            agent_ds[start_idx:end_idx] = np.transpose(agent_data_to_save, (1, 0, 2))
            obj_ds[start_idx:end_idx] = np.transpose(obj_data_to_save, (1, 0, 2))

            random_saved += num_this_batch
            total_episodes_saved += num_this_batch
            print(f"  ...Collected random episodes: {random_saved}/{NUM_RANDOM}")

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")