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
from .go2_env import Go2Env  # Make sure you have access to your modified Go2Env
import faiss
import logging


class ICMModule(nn.Module):
    # ... (ICMModule definition remains the same) ...
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
        # To train the encoder, gradients should only come from the inverse model loss.
        # Detach the target for the forward model loss to prevent gradients from flowing back.
        state_action = torch.cat([state, action], dim=-1)
        phi = self.encoder(state_action)
        with torch.no_grad():
            phi_next = self.encoder(torch.cat([next_state, action], dim=-1))

        # Forward prediction loss (measures predictability of the next state)
        pred_phi_next = self.forward_model(torch.cat([phi, action], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=-1)

        # Inverse prediction loss (measures understanding of dynamics)
        # Re-compute phi_next with gradients for the inverse model path
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

    # IMPORTANT: Update your obs_cfg to reflect the new observation size (e.g., 45 -> 51)
    obs_cfg["num_obs"] += 6

    original_entropy = train_cfg['algorithm'].get('entropy_coef', 0.0)
    train_cfg['algorithm']['entropy_coef'] = 0.02
    print(
        f"Updated entropy coefficient from {original_entropy} to {train_cfg['algorithm']['entropy_coef']} to encourage exploration.")

    gs.init()
    logging.getLogger('genesis').setLevel(logging.WARNING)

    reward_cfg["reward_scales"] = {"survive": 1.0, "termination": -200.0}

    env = Go2Env(num_envs=NUM_ENVS, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
                 show_viewer=NUM_ENVS < 128)

    # Re-initialize the runner with the modified configs
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    policy_alg = runner.alg
    icm = ICMModule(state_dim=env.num_obs, action_dim=env.num_actions, hidden_dim=POLICY_HIDDEN_DIM).to(gs.device)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-4)

    n_dofs = len(env.motors_dof_idx)
    input_dim = env.num_obs
    output_dim = env.num_actions
    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)

    # Buffers for agent and object trajectories, kept on GPU for performance
    agent_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs), device=gs.device)
    obj_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, 3), device=gs.device)  # ADDED: Buffer for ball position

    embedding_dim = (max_episode_len // DOWNSAMPLE_FACTOR) * n_dofs
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)

    total_episodes_saved = 0
    mean_surrogate_loss = 0
    icm_loss = torch.tensor(0.0)
    agent_save_buffer = []  # Buffer for agent trajectories
    obj_save_buffer = []  # Buffer for object trajectories

    print(f"Starting curiosity-driven data generation. Target: {EPISODES_TO_COLLECT} episodes.")

    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()
    obs, _ = env.reset()

    with h5py.File(SAVE_FILENAME, 'w') as f:
        agent_ds = f.create_dataset('agent_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                    dtype='float32', compression="gzip")
        obj_ds = f.create_dataset('obj_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, 3), dtype='float32',
                                  compression="gzip")

        episodes_since_last_save = 0
        while total_episodes_saved < EPISODES_TO_COLLECT:
            # --- A. Collect a batch of episodes on the GPU ---
            for t in range(max_episode_len):
                actions = policy_alg.act(obs, obs)
                next_obs, rews, dones, infos = env.step(actions)
                with torch.no_grad():
                    intrinsic_reward, _ = icm(obs, actions, next_obs)
                total_reward = rews + intrinsic_reward * CURIOSITY_BETA
                policy_alg.process_env_step(total_reward, dones, infos)

                # Record trajectories for both agent and object
                agent_traj_buffer[t] = env.robot.get_dofs_position(env.motors_dof_idx)
                obj_traj_buffer[t] = env.ball.get_pos()  # ADDED: Record ball position

                obs = next_obs

                if policy_alg.storage.step >= runner.num_steps_per_env:
                    num_transitions = policy_alg.storage.step
                    if num_transitions > 1:
                        all_states, all_actions = policy_alg.storage.observations.view(-1,
                                                                                       input_dim), policy_alg.storage.actions.view(
                            -1, output_dim)
                        end_idx = (num_transitions - 1) * policy_alg.storage.num_envs
                        batch_states, batch_actions = all_states[:end_idx], all_actions[:end_idx]
                        start_idx_next, end_idx_next = policy_alg.storage.num_envs, num_transitions * policy_alg.storage.num_envs
                        batch_next_states = all_states[start_idx_next:end_idx_next]
                        f_loss, i_loss = icm(batch_states, batch_actions, batch_next_states)
                        icm_loss = f_loss.mean() + i_loss.mean()
                        icm_optimizer.zero_grad()
                        icm_loss.backward()
                        icm_optimizer.step()
                    policy_alg.compute_returns(obs)
                    mean_value_loss, mean_surrogate_loss, _, _, _ = policy_alg.update()

            # --- B. Transfer data to CPU and Deduplicate ---
            agent_data_batch_numpy = agent_traj_buffer.cpu().numpy()
            obj_data_batch_numpy = obj_traj_buffer.cpu().numpy()  # ADDED

            agent_data_batch = np.transpose(agent_data_batch_numpy, (1, 0, 2))
            obj_data_batch = np.transpose(obj_data_batch_numpy, (1, 0, 2))  # ADDED

            downsampled_batch = agent_data_batch[:, ::DOWNSAMPLE_FACTOR, :]
            new_embeddings = downsampled_batch.reshape(NUM_ENVS, -1).astype(np.float32)
            faiss.normalize_L2(new_embeddings)

            if faiss_index.ntotal == 0:
                unique_indices = np.arange(len(new_embeddings))
            else:
                D, I = faiss_index.search(x=new_embeddings, k=1)
                is_unique_mask = D[:, 0] < SIMILARITY_THRESHOLD
                unique_indices = np.where(is_unique_mask)[0]

            if unique_indices.size > 0:
                agent_save_buffer.append(agent_data_batch[unique_indices])  # CHANGED
                obj_save_buffer.append(obj_data_batch[unique_indices])  # ADDED
                faiss_index.add(x=new_embeddings[unique_indices])
                episodes_since_last_save += len(unique_indices)

            if episodes_since_last_save >= WRITE_BUFFER_SIZE or total_episodes_saved + episodes_since_last_save >= EPISODES_TO_COLLECT:
                if agent_save_buffer:
                    # Concatenate and write agent data
                    agent_block_to_write = np.concatenate(agent_save_buffer, axis=0)
                    num_in_block = agent_block_to_write.shape[0]
                    start_idx, end_idx = total_episodes_saved, min(total_episodes_saved + num_in_block,
                                                                   EPISODES_TO_COLLECT)
                    num_to_write = end_idx - start_idx
                    if num_to_write > 0:
                        print(f"  ...Writing {num_to_write} episodes to disk...")
                        agent_ds[start_idx:end_idx] = agent_block_to_write[:num_to_write]

                        # Concatenate and write corresponding object data
                        obj_block_to_write = np.concatenate(obj_save_buffer, axis=0)
                        obj_ds[start_idx:end_idx] = obj_block_to_write[:num_to_write]

                        total_episodes_saved = end_idx

                    agent_save_buffer, obj_save_buffer = [], []
                    episodes_since_last_save = 0

            print(
                f"  ...Collected: {total_episodes_saved}/{EPISODES_TO_COLLECT} | PPO Loss: {mean_surrogate_loss:.3f} | ICM Loss: {icm_loss.item():.3f}")

        # Final write for any remaining items in the buffer
        if agent_save_buffer:
            agent_block_to_write = np.concatenate(agent_save_buffer, axis=0)
            obj_block_to_write = np.concatenate(obj_save_buffer, axis=0)
            num_in_block = agent_block_to_write.shape[0]
            start_idx, end_idx = total_episodes_saved, min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
            num_to_write = end_idx - start_idx
            if num_to_write > 0:
                print(f"  ...Writing final {num_to_write} episodes to disk...")
                agent_ds[start_idx:end_idx] = agent_block_to_write[:num_to_write]
                obj_ds[start_idx:end_idx] = obj_block_to_write[:num_to_write]
                total_episodes_saved = end_idx

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")