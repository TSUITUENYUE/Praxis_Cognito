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
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque


class ICMModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ICMModule, self).__init__()
        # Encoder: Maps state-action to feature space
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        # Forward model: Predicts next state features
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        # Inverse model: Predicts action from state transitions
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, action, next_state):
        # Encode state and next_state
        state_action = torch.cat([state, action], dim=-1)
        phi = self.encoder(state_action)
        phi_next = self.encoder(torch.cat([next_state, action], dim=-1))

        # Forward prediction
        pred_phi_next = self.forward_model(torch.cat([phi, action], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=-1)

        # Inverse prediction
        pred_action = self.inverse_model(torch.cat([phi, phi_next], dim=-1))
        inverse_loss = F.mse_loss(pred_action, action, reduction='none').mean(dim=-1)

        return forward_loss, inverse_loss


def compute_trajectory_embedding(traj, downsample_factor=5):
    """Compute a compact embedding for deduplication."""
    # Downsample and flatten trajectory (e.g., joint positions)
    traj_downsampled = traj[::downsample_factor].reshape(-1)
    return traj_downsampled


def generate(cfg: DictConfig):
    # Data Gen Configs
    NUM_ENVS = cfg.dataset.num_envs
    EPISODES_TO_COLLECT = cfg.dataset.episodes  # Target: 500,000
    MAX_EPISODE_SECONDS = cfg.dataset.max_episode_seconds  # 5s
    FRAME_RATE = cfg.dataset.frame_rate  # 30 FPS
    AGENT = cfg.agent.name
    path = f"./Pretrain/data/{AGENT}/{NUM_ENVS} {EPISODES_TO_COLLECT} {MAX_EPISODE_SECONDS} {FRAME_RATE}"
    os.makedirs(path, exist_ok=True)
    SAVE_FILENAME = f"{path}/{NUM_ENVS} {EPISODES_TO_COLLECT} {MAX_EPISODE_SECONDS} {FRAME_RATE}.h5"

    CURIOSITY_BETA = cfg.dataset.curiosity_beta  # Weight for intrinsic reward
    POLICY_HIDDEN_DIM = cfg.dataset.policy_hidden_dim
    SIMILARITY_THRESHOLD = cfg.dataset.similarity_threshold  # For deduplication

    log_dir = cfg.dataset.log_dir
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    gs.init()
    reward_cfg["reward_scales"] = {
        "tracking_lin_vel": 1.0,  # Encourage velocity tracking
        "tracking_ang_vel": 1.0,  # Encourage yaw tracking
        "similar_to_default": -0.5,  # Penalize deviation from default pose
        "base_height": -0.5  # Penalize height deviation
    }

    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=NUM_ENVS < 128,
    )

    # Initialize PPO policy and ICM
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    policy = runner.get_inference_policy(device=gs.device)  # Base PPO policy
    icm = ICMModule(state_dim=env.num_obs, action_dim=env.num_actions, hidden_dim=POLICY_HIDDEN_DIM).to(gs.device)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

    # Joint/DoF from env
    n_dofs = len(env.motors_dof_idx)
    input_dim = env.num_obs
    output_dim = env.num_actions
    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)

    # Buffers
    agent_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs), device=gs.device)
    obj_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, 3), device=gs.device)
    state_buffer = torch.zeros((max_episode_len, NUM_ENVS, input_dim), device=gs.device)
    action_buffer = torch.zeros((max_episode_len, NUM_ENVS, output_dim), device=gs.device)
    next_state_buffer = torch.zeros((max_episode_len, NUM_ENVS, input_dim), device=gs.device)

    # Deduplication buffer
    traj_embeddings = deque(maxlen=250000)  # Increased for 500k episodes

    total_episodes_saved = 0
    print(f"Starting curiosity-driven data generation on flat terrain. Target: {EPISODES_TO_COLLECT} episodes.")

    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()

    with h5py.File(SAVE_FILENAME, 'w') as f:
        agent_ds = f.create_dataset('agent_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs), dtype='float32', compression="gzip")
        obj_ds = f.create_dataset('obj_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, 3), dtype='float32', compression="gzip")

        # Pre-train ICM briefly to bootstrap curiosity
        print("Pre-training ICM...")
        for pretrain_iter in range(1000):  # Short pre-training
            obs, _ = env.reset()

            # --- NO LOSS CALCULATION IN THIS LOOP ---
            # This loop is now only for collecting a full episode of transitions.
            with torch.no_grad():  # Actions are treated as data for ICM, no need for policy grads here
                for t in range(max_episode_len):
                    state_buffer[t] = obs
                    actions = policy(obs)
                    action_buffer[t] = actions
                    obs, rews, dones, infos = env.step(actions)
                    next_state_buffer[t] = obs

            # --- BATCH LOSS CALCULATION AFTER THE EPISODE ---

            # Reshape the buffers into a single large batch of transitions
            # Shape goes from (time, envs, dim) to (time * envs, dim)
            s = state_buffer.reshape(-1, input_dim)
            a = action_buffer.reshape(-1, output_dim)
            s_next = next_state_buffer.reshape(-1, input_dim)

            # Now, compute the loss on the entire batch of transitions at once
            forward_loss, inverse_loss = icm(s, a, s_next)

            # The final loss is a simple mean over the batch dimension
            total_loss = forward_loss.mean() + inverse_loss.mean()

            # Single backward pass on the cleanly computed total loss
            icm_optimizer.zero_grad()
            total_loss.backward()
            icm_optimizer.step()
            print(f"  ...Pre-training iteration {pretrain_iter + 1}/1000, Loss: {total_loss.item():.4f}")

        print("Pre-training complete. Starting episode collection...")

        # Curiosity-driven collection on flat terrain
        while total_episodes_saved < EPISODES_TO_COLLECT:
            obs, _ = env.reset()
            valid_episodes = torch.ones(NUM_ENVS, dtype=torch.bool, device=gs.device)

            with torch.no_grad():
                for t in range(max_episode_len):
                    state_buffer[t] = obs
                    actions = policy(obs)
                    action_buffer[t] = actions
                    obs, rews, dones, infos = env.step(actions)
                    next_state_buffer[t] = obs

                    # Compute intrinsic reward (curiosity)
                    forward_loss, _ = icm(state_buffer[t], actions, next_state_buffer[t])
                    intrinsic_rewards = forward_loss * CURIOSITY_BETA

                    # Joint limit check
                    joint_positions = env.robot.get_dofs_position(env.motors_dof_idx)  # Shape: (NUM_ENVS, n_dofs)

                    agent_traj_buffer[t] = joint_positions
                    obj_traj_buffer[t] = torch.zeros((NUM_ENVS, 3), device=gs.device)

            # Deduplicate and save valid trajectories
            start_idx = total_episodes_saved
            num_this_batch = min(NUM_ENVS, EPISODES_TO_COLLECT - total_episodes_saved)
            valid_indices = torch.where(valid_episodes)[0][:num_this_batch]

            if valid_indices.size(0) > 0:
                agent_data = agent_traj_buffer[:, valid_indices, :].cpu().numpy()
                obj_data = obj_traj_buffer[:, valid_indices, :].cpu().numpy()

                # Deduplication
                valid_batch = []
                for i in range(agent_data.shape[1]):
                    traj_embedding = compute_trajectory_embedding(agent_data[:, i, :])
                    is_unique = True
                    for existing_emb in traj_embeddings:
                        if cosine_similarity([traj_embedding], [existing_emb])[0, 0] > SIMILARITY_THRESHOLD:
                            is_unique = False
                            break
                    if is_unique:
                        valid_batch.append(i)
                        traj_embeddings.append(traj_embedding)

                if valid_batch:
                    valid_batch = valid_batch[:num_this_batch]  # Ensure we don't overshoot
                    agent_data_to_save = agent_data[:, valid_batch, :]
                    obj_data_to_save = obj_data[:, valid_batch, :]

                    end_idx = start_idx + len(valid_batch)
                    if end_idx > start_idx:
                        agent_ds[start_idx:end_idx] = np.transpose(agent_data_to_save, (1, 0, 2))
                        obj_ds[start_idx:end_idx] = np.transpose(obj_data_to_save, (1, 0, 2))
                        total_episodes_saved += len(valid_batch)
                        print(f"  ...Collected curiosity-driven episodes: {total_episodes_saved}/{EPISODES_TO_COLLECT}")

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")
