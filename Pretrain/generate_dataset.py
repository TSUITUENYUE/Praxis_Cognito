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

# ============================
# [CHANGE: ICM is state-only]
# ============================
class ICMModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ICMModule, self).__init__()
        latent_dim = hidden_dim // 2
        # state-only encoder (no action leakage)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # forward: condition on action
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # inverse: infer action from (phi(s), phi(s'))
        self.inverse_model = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, action, next_state):
        # encode states only
        phi_s  = self.encoder(state)
        phi_sp = self.encoder(next_state)
        # forward loss uses stop-grad on target
        pred_phi_sp = self.forward_model(torch.cat([phi_s, action], dim=-1))
        forward_loss = F.mse_loss(pred_phi_sp, phi_sp.detach(), reduction='none').mean(dim=-1)
        # inverse uses gradients on both encodings
        pred_action = self.inverse_model(torch.cat([phi_s, phi_sp], dim=-1))
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

    # original_entropy = train_cfg['algorithm'].get('entropy_coef', 0.0)
    train_cfg['algorithm']['entropy_coef'] = 0.02

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
    primitives = os.path.join(log_dir, "model_199.pt")
    runner.load(primitives)
    policy_alg = runner.alg

    # ICM (curiosity)
    icm = ICMModule(state_dim=env.num_obs, action_dim=env.num_actions, hidden_dim=POLICY_HIDDEN_DIM).to(gs.device)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-4)

    # RMS for ICM (kept exactly as you had it)
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

    # [CHANGE: store policy mean & std (not samples); align ICM with exec_actions]
    act_mean_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, env.num_actions), device=gs.device)
    act_std_traj_buffer  = torch.zeros((max_episode_len, NUM_ENVS, env.num_actions), device=gs.device)

    # keep applied (clipped+latency, pre-scale) actions for ICM alignment only
    exec_traj_buffer     = torch.zeros((max_episode_len, NUM_ENVS, env.num_actions), device=gs.device)

    dof_vel_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)
    done_traj_buffer    = torch.zeros((max_episode_len, NUM_ENVS), dtype=torch.bool, device=gs.device)

    # >>> DEDUP EMBEDDING BACK TO JOINTS-ONLY <<<
    embedding_dim = (max_episode_len // DOWNSAMPLE_FACTOR) * n_dofs
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)

    total_episodes_saved = 0
    mean_surrogate_loss = 0
    icm_loss = torch.tensor(0.0)
    agent_save_buffer, obj_save_buffer = [], []
    obs_save_buffer, act_mean_save_buffer, act_std_save_buffer, vel_save_buffer = [], [], [], []
    valid_len_save_buffer = []

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

    # Save normalizers for downstream reuse (UNCHANGED per your request)
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
        # [CHANGE: replace 'actions' dataset with mean & std datasets]
        action_means_ds = f.create_dataset('action_means',
                                           shape=(EPISODES_TO_COLLECT, max_episode_len, env.num_actions),
                                           dtype='float32', compression="gzip")
        action_stds_ds  = f.create_dataset('action_stds',
                                           shape=(EPISODES_TO_COLLECT, max_episode_len, env.num_actions),
                                           dtype='float32', compression="gzip")
        dof_vel_ds = f.create_dataset('dof_vel',
                                      shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                      dtype='float32', compression="gzip")
        valid_len_ds = f.create_dataset('valid_length',
                                        shape=(EPISODES_TO_COLLECT,), dtype='int32', compression="gzip")

        # Metadata (UNCHANGED except we do NOT add extra normalizer states)
        meta = f.create_group('meta')
        meta.attrs['dt'] = float(env.dt)
        act_scale = env.env_cfg.get('action_scale', 0.25)
        if isinstance(act_scale, (list, tuple, np.ndarray, torch.Tensor)):
            meta.create_dataset('action_scale', data=np.asarray(act_scale, dtype=np.float32))
        else:
            meta.attrs['action_scale'] = float(act_scale)
        meta.create_dataset('default_dof_pos', data=np.asarray(env.default_dof_pos.cpu(), dtype=np.float32))
        meta.attrs['normalizer_pt'] = os.path.abspath(norm_state_path)
        meta.attrs['faiss_threshold'] = float(SIMILARITY_THRESHOLD)
        meta.attrs['downsample_factor'] = int(DOWNSAMPLE_FACTOR)

        episodes_since_last_save = 0
        while total_episodes_saved < EPISODES_TO_COLLECT:
            f_loss = torch.tensor(0.0, device=gs.device)
            i_loss = torch.tensor(0.0, device=gs.device)

            first_done_seen = torch.full((NUM_ENVS,), fill_value=max_episode_len, dtype=torch.int32, device=gs.device)
            for t in range(max_episode_len):
                # --- PRE-STEP SNAPSHOT (ALIGN INDICES) --------------------------
                # store obs_t for decoder/teacher pairing
                obs_traj_buffer[t] = obs

                # [NEW: move kinematics/object logging BEFORE env.step]
                agent_traj_buffer[t] = env.dof_pos  # q_t  (shape [E, DoF])
                dof_vel_traj_buffer[t] = env.dof_vel  # qdot_t
                if hasattr(env, "ball"):
                    obj_traj_buffer[t] = env.ball.get_pos()  # object pos at t
                # ----------------------------------------------------------------

                # act with normalized obs like training (teacher at time t)
                actions = policy_alg.act(obs_n, critic_obs_n)

                # record teacher distribution (per your request: mean & std only)
                act_mean = policy_alg.actor_critic.action_mean.detach()
                act_std = policy_alg.actor_critic.action_std.detach()
                # ensure shapes are [E, A] for storage
                if act_mean.dim() == 1:
                    act_mean = act_mean.unsqueeze(0).expand(NUM_ENVS, -1)
                elif act_mean.size(0) == 1 and NUM_ENVS > 1:
                    act_mean = act_mean.expand(NUM_ENVS, -1)
                if act_std.dim() == 1:
                    act_std = act_std.unsqueeze(0).expand(NUM_ENVS, -1)
                elif act_std.size(0) == 1 and NUM_ENVS > 1:
                    act_std = act_std.expand(NUM_ENVS, -1)

                act_mean_traj_buffer[t] = act_mean
                act_std_traj_buffer[t] = act_std

                # --- ENV STEP (applies a_t, advances to t+1) --------------------
                next_obs, rews, dones, infos = env.step(actions)
                done_traj_buffer[t] = (dones.squeeze(-1) > 0) if dones.ndim == 2 else (dones > 0)

                # keep aligned applied actions (clipped + latency, pre-scale) for ICM at index t
                exec_traj_buffer[t] = env.exec_actions
                # ----------------------------------------------------------------

                # [REMOVED: post-step kinematics logging]
                # q_now  = env.robot.get_dofs_position(env.motors_dof_idx)
                # dq_now = env.robot.get_dofs_velocity(env.motors_dof_idx)
                # agent_traj_buffer[t]   = q_now
                # dof_vel_traj_buffer[t] = dq_now
                # if hasattr(env, "ball"):
                #     obj_traj_buffer[t] = env.ball.get_pos()

                # curiosity plumbing (unchanged; aligned to applied a_t)
                state_normalizer.update(obs)
                action_normalizer.update(env.exec_actions)
                state_normalizer.update(next_obs)
                norm_obs = state_normalizer.normalize(obs)
                norm_actions = action_normalizer.normalize(env.exec_actions)
                norm_next_obs = state_normalizer.normalize(next_obs)
                with torch.no_grad():
                    forward_loss_step, inverse_loss_step = icm(norm_obs, norm_actions, norm_next_obs)
                    intrinsic_reward = forward_loss_step
                reward_normalizer.update(intrinsic_reward.unsqueeze(-1))
                intrinsic_reward /= torch.sqrt(reward_normalizer.var + reward_normalizer.epsilon)
                total_reward = rews + intrinsic_reward * CURIOSITY_BETA

                # PPO cadence (unchanged)
                policy_alg.process_env_step(total_reward, dones, infos)

                # advance to next pre-step obs
                obs = next_obs
                critic_obs = infos["observations"]["critic"] if (
                            "observations" in infos and "critic" in infos["observations"]) else obs
                obs_n = runner.obs_normalizer(obs)
                critic_obs_n = runner.critic_obs_normalizer(critic_obs)

                # first done per env (unchanged)
                freshly_done = (done_traj_buffer[t] & (first_done_seen == max_episode_len))
                if freshly_done.any():
                    idxs = torch.nonzero(freshly_done, as_tuple=False).squeeze(-1)
                    first_done_seen[idxs] = t + 1  # inclusive

                # ICM minibatch using our aligned buffers (unchanged)
                if policy_alg.storage.step >= runner.num_steps_per_env:
                    K = policy_alg.storage.step
                    if K > 1:
                        start = t - K + 1
                        if start >= 0:
                            s = obs_traj_buffer[start: start + K - 1].reshape(-1, input_dim)
                            a = exec_traj_buffer[start: start + K - 1].reshape(-1, output_dim)
                            s2 = obs_traj_buffer[start + 1: start + K].reshape(-1, input_dim)

                            f_loss, i_loss = icm(
                                state_normalizer.normalize(s),
                                action_normalizer.normalize(a),
                                state_normalizer.normalize(s2),
                            )
                            icm_loss = f_loss.mean() + i_loss.mean()
                            icm_optimizer.zero_grad()
                            icm_loss.backward()
                            icm_optimizer.step()

                    policy_alg.compute_returns(critic_obs_n)
                    mean_value_loss, mean_surrogate_loss, _, _, _ = policy_alg.update()

            # simple last-step telemetry (mean over env dim)
            action_mean_scalar = act_mean.mean().item()
            action_std_scalar  = act_std.mean().item()

            # ---- Build dedup embeddings (JOINTS ONLY, NO Z-SCORE) ----
            agent_np = agent_traj_buffer.cpu().numpy()    # [T, E, DoF]
            obj_np   = obj_traj_buffer.cpu().numpy()      # [T, E, 3]
            obs_np   = obs_traj_buffer.cpu().numpy()
            act_mean_np = act_mean_traj_buffer.cpu().numpy()
            act_std_np  = act_std_traj_buffer.cpu().numpy()
            vel_np   = dof_vel_traj_buffer.cpu().numpy()
            valid_len_np = np.where(first_done_seen.cpu().numpy() == max_episode_len,
                                    max_episode_len,
                                    first_done_seen.cpu().numpy())

            # [E, T, ...]
            agent_batch = np.transpose(agent_np, (1, 0, 2))
            obj_batch   = np.transpose(obj_np,   (1, 0, 2))
            obs_batch   = np.transpose(obs_np,   (1, 0, 2))
            act_mean_batch = np.transpose(act_mean_np, (1, 0, 2))
            act_std_batch  = np.transpose(act_std_np,  (1, 0, 2))
            vel_batch   = np.transpose(vel_np,   (1, 0, 2))

            # zero invalid tail (matches written data)
            for e in range(NUM_ENVS):
                L = valid_len_np[e]
                if L < max_episode_len:
                    agent_batch[e, L:]    = 0.0
                    vel_batch[e,   L:]    = 0.0
                    obj_batch[e,   L:]    = 0.0
                    obs_batch[e,   L:]    = 0.0
                    act_mean_batch[e, L:] = 0.0
                    act_std_batch[e,  L:] = 0.0

            # Downsample joints ONLY for dedup
            agent_dsmp = agent_batch[:, ::DOWNSAMPLE_FACTOR, :]  # [E, Td, DoF]

            new_embeddings = agent_dsmp.reshape(NUM_ENVS, -1).astype(np.float32)
            faiss.normalize_L2(new_embeddings)  # cosine via IP

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
                act_mean_save_buffer.append(act_mean_batch[unique_indices])
                act_std_save_buffer.append(act_std_batch[unique_indices])
                vel_save_buffer.append(vel_batch[unique_indices])
                valid_len_save_buffer.append(valid_len_np[unique_indices])
                faiss_index.add(x=new_embeddings[unique_indices])
                episodes_since_last_save += len(unique_indices)

            # Write in blocks
            if episodes_since_last_save >= WRITE_BUFFER_SIZE or total_episodes_saved + episodes_since_last_save >= EPISODES_TO_COLLECT:
                if agent_save_buffer:
                    agent_block    = np.concatenate(agent_save_buffer, axis=0)
                    obj_block      = np.concatenate(obj_save_buffer,   axis=0)
                    obs_block      = np.concatenate(obs_save_buffer,   axis=0)
                    act_mean_block = np.concatenate(act_mean_save_buffer, axis=0)
                    act_std_block  = np.concatenate(act_std_save_buffer,  axis=0)
                    vel_block      = np.concatenate(vel_save_buffer,   axis=0)
                    vlen_block     = np.concatenate(valid_len_save_buffer, axis=0)

                    num_in_block = agent_block.shape[0]
                    start_idx = total_episodes_saved
                    end_idx = min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
                    num_to_write = end_idx - start_idx
                    if num_to_write > 0:
                        print(f"  ...Writing {num_to_write} episodes to disk...")
                        agent_ds[start_idx:end_idx]       = agent_block[:num_to_write]
                        obj_ds[start_idx:end_idx]         = obj_block[:num_to_write]
                        obs_ds[start_idx:end_idx]         = obs_block[:num_to_write]
                        action_means_ds[start_idx:end_idx]= act_mean_block[:num_to_write]
                        action_stds_ds[start_idx:end_idx] = act_std_block[:num_to_write]
                        dof_vel_ds[start_idx:end_idx]     = vel_block[:num_to_write]
                        valid_len_ds[start_idx:end_idx]   = vlen_block[:num_to_write]
                        total_episodes_saved = end_idx

                    agent_save_buffer, obj_save_buffer = [], []
                    obs_save_buffer, act_mean_save_buffer, act_std_save_buffer, vel_save_buffer = [], [], [], []
                    valid_len_save_buffer = []
                    episodes_since_last_save = 0

            print(f"  ...Collected: {total_episodes_saved}/{EPISODES_TO_COLLECT} | PPO Loss: {mean_surrogate_loss:.3f} "
                  f"| Fwd Loss: {f_loss.mean().item():.3f} | Inv Loss: {i_loss.mean().item():.3f} "
                  f"| Action Mean: {action_mean_scalar:.3f} | Action Std: {action_std_scalar:.3f}")

        # Final flush if remaining
        if agent_save_buffer:
            agent_block    = np.concatenate(agent_save_buffer, axis=0)
            obj_block      = np.concatenate(obj_save_buffer,   axis=0)
            obs_block      = np.concatenate(obs_save_buffer,   axis=0)
            act_mean_block = np.concatenate(act_mean_save_buffer, axis=0)
            act_std_block  = np.concatenate(act_std_save_buffer,  axis=0)
            vel_block      = np.concatenate(vel_save_buffer,   axis=0)
            vlen_block     = np.concatenate(valid_len_save_buffer, axis=0)

            num_in_block = agent_block.shape[0]
            start_idx = total_episodes_saved
            end_idx = min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
            num_to_write = end_idx - start_idx
            if num_to_write > 0:
                print(f"  ...Writing final {num_to_write} episodes to disk...")
                agent_ds[start_idx:end_idx]        = agent_block[:num_to_write]
                obj_ds[start_idx:end_idx]          = obj_block[:num_to_write]
                obs_ds[start_idx:end_idx]          = obs_block[:num_to_write]
                action_means_ds[start_idx:end_idx] = act_mean_block[:num_to_write]
                action_stds_ds[start_idx:end_idx]  = act_std_block[:num_to_write]
                dof_vel_ds[start_idx:end_idx]      = vel_block[:num_to_write]
                valid_len_ds[start_idx:end_idx]    = vlen_block[:num_to_write]
                total_episodes_saved = end_idx

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Normalizers saved to '{norm_state_path}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")
