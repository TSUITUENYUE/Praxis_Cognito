import genesis as gs
import numpy as np
import time
import torch
import h5py
import os
from omegaconf import OmegaConf,DictConfig
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from .go2_env_primitives import Go2Env
from Model.agent import Agent
import faiss
from torch.cuda.amp import autocast, GradScaler  # <-- AMP

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
            nn.Linear(state_dim, hidden_dim),
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
        phi = self.encoder(state)
        phi_next_with_grad = self.encoder(next_state)
        with torch.no_grad():
            phi_next = phi_next_with_grad.detach()
        pred_phi_next = self.forward_model(torch.cat([phi, action], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=-1)
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

    #env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg.rl

    # original_entropy = train_cfg['algorithm'].get('entropy_coef', 0.0)
    cfg.rl.train.algorithm.entropy_coef = 0.02

    gs.init(logging_level='warning')

    cfg.rl.reward["reward_scales"] = {"survive": 1.0, "termination": -200.0}
    agent = Agent(**cfg.agent)
    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=cfg.rl.env, obs_cfg=cfg.rl.obs, reward_cfg=cfg.rl.reward, command_cfg=cfg.rl.command,
        show_viewer=NUM_ENVS < 128,
        agent=agent
    )
    train_cfg = OmegaConf.to_container(cfg.rl.train, resolve=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    primitives = os.path.join(log_dir, "model_200.pt")
    runner.load(primitives)
    policy_alg = runner.alg

    # ICM (curiosity)
    icm = ICMModule(state_dim=env.num_obs, action_dim=env.num_actions, hidden_dim=POLICY_HIDDEN_DIM).to(gs.device)

    icm = torch.compile(icm)  # torch 2.x
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-4)

    # AMP scaler (ICM only; PPO uses library's internal optimizers)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # RMS for ICM
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
    dof_pos_traj_buffer   = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)
    dof_vel_traj_buffer   = torch.zeros((max_episode_len, NUM_ENVS, n_dofs),      device=gs.device)

    base_pos_buffer       = torch.zeros((max_episode_len, NUM_ENVS, 3),           device=gs.device)
    base_vel_buffer       = torch.zeros((max_episode_len, NUM_ENVS, 3),           device=gs.device)
    base_ang_buffer       = torch.zeros((max_episode_len, NUM_ENVS, 3),           device=gs.device)

    ball_pos_buffer     = torch.zeros((max_episode_len, NUM_ENVS, 3), device=gs.device)  # NEW (ball)
    ball_vel_buffer     = torch.zeros((max_episode_len, NUM_ENVS, 3), device=gs.device)  # NEW (ball)
    ball_ang_buffer     = torch.zeros((max_episode_len, NUM_ENVS, 3), device=gs.device)  # NEW (ball)

    obs_traj_buffer       = torch.zeros((max_episode_len, NUM_ENVS, env.num_obs), device=gs.device)
    act_traj_buffer       = torch.zeros((max_episode_len, NUM_ENVS, env.num_actions), device=gs.device)

    done_traj_buffer      = torch.zeros((max_episode_len, NUM_ENVS), dtype=torch.bool, device=gs.device)

    # >>> DEDUP EMBEDDING BACK TO JOINTS-ONLY <<<
    embedding_dim = (max_episode_len // DOWNSAMPLE_FACTOR) * (n_dofs + 2 * 3)
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)

    total_episodes_saved = 0
    mean_surrogate_loss = 0
    icm_loss = torch.tensor(0.0)

    dof_pos_save_buffer,dof_vel_save_buffer, = [], []
    base_pos_save_buffer,base_vel_save_buffer,base_ang_save_buffer = [], [], []
    ball_pos_save_buffer,ball_vel_save_buffer,ball_ang_save_buffer = [], [], []
    obs_save_buffer, act_save_buffer= [], []
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

    # Save normalizers for downstream reuse
    norm_state_path = os.path.join(path, "normalizers.pt")
    torch.save({
        "obs_norm": runner.obs_normalizer.state_dict(),
        "critic_obs_norm": runner.critic_obs_normalizer.state_dict(),
    }, norm_state_path)
    print(f"Saved normalizer state to {norm_state_path}")

    with h5py.File(SAVE_FILENAME, 'w') as f:
        dof_pos_ds = f.create_dataset('dof_pos',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                    dtype='float32', compression="gzip")
        dof_vel_ds = f.create_dataset('dof_vel',
                                      shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                      dtype='float32', compression="gzip")

        base_pos_ds = f.create_dataset('base_pos',
                                       shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                       dtype='float32', compression="gzip")
        base_vel_ds = f.create_dataset('base_vel',
                                       shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                       dtype='float32', compression="gzip")
        base_ang_ds = f.create_dataset('base_ang',
                                         shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                         dtype='float32', compression="gzip")
        ball_pos_ds = f.create_dataset('ball_pos',                                           # NEW (ball)
                                       shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                       dtype='float32', compression="gzip")
        ball_vel_ds = f.create_dataset('ball_vel',                                           # NEW (ball)
                                       shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                       dtype='float32', compression="gzip")
        ball_ang_ds = f.create_dataset('ball_ang',                                           # NEW (ball)
                                       shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                       dtype='float32', compression="gzip")

        obs_ds   = f.create_dataset('obs',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, env.num_obs),
                                    dtype='float32', compression="gzip")
        act_ds = f.create_dataset('act',
                                      shape=(EPISODES_TO_COLLECT, max_episode_len, env.num_actions),
                                      dtype='float32', compression="gzip")

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
        meta.create_dataset('default_dof_pos_', data=np.asarray(env.default_dof_pos.cpu(), dtype=np.float32))
        meta.attrs['normalizer_pt'] = os.path.abspath(norm_state_path)
        meta.attrs['faiss_threshold'] = float(SIMILARITY_THRESHOLD)
        meta.attrs['downsample_factor'] = int(DOWNSAMPLE_FACTOR)

        episodes_since_last_save = 0
        while total_episodes_saved < EPISODES_TO_COLLECT:
            f_loss = torch.tensor(0.0, device=gs.device)
            i_loss = torch.tensor(0.0, device=gs.device)
            action_mean = 0.0
            action_std = 0.0

            first_done_seen = torch.full((NUM_ENVS,), fill_value=max_episode_len, dtype=torch.int32, device=gs.device)
            for t in range(max_episode_len):
                # ================== ROLLOUT (no autograd, AMP for speed) ==================
                with torch.no_grad():
                    obs_traj_buffer[t] = obs

                    raw_act = policy_alg.act(obs_n, critic_obs_n)
                    act = torch.clamp(raw_act, -100.0, 100.0)   # executed action
                    act_traj_buffer[t] = act

                    next_obs, rews, dones, infos = env.step(act)
                    done_traj_buffer[t] = (dones.squeeze(-1) > 0) if dones.ndim == 2 else (dones > 0)

                    q_now  = env.robot.get_dofs_position(env.motors_dof_idx)
                    dq_now = env.robot.get_dofs_velocity(env.motors_dof_idx)
                    dof_pos_traj_buffer[t] = q_now
                    dof_vel_traj_buffer[t] = dq_now

                    base_pos_buffer[t] = env.robot.get_pos()
                    base_vel_buffer[t] = env.robot.get_vel()
                    base_ang_buffer[t] = env.robot.get_ang()

                    ball_pos_buffer[t] = env.ball.get_pos()
                    ball_vel_buffer[t] = env.ball.get_vel()
                    ball_ang_buffer[t] = env.ball.get_ang()

                    state_normalizer.update(obs)
                    action_normalizer.update(act)
                    state_normalizer.update(next_obs)
                    norm_obs, norm_act, norm_next_obs = (
                        state_normalizer.normalize(obs),
                        action_normalizer.normalize(act),
                        state_normalizer.normalize(next_obs),
                    )

                    # AMP autocast for ICM forward (no grad)
                    with autocast(enabled=torch.cuda.is_available()):
                        intrinsic_reward, _ = icm(norm_obs, norm_act, norm_next_obs)

                    reward_normalizer.update(intrinsic_reward.unsqueeze(-1))
                    intrinsic_reward /= torch.sqrt(reward_normalizer.var + reward_normalizer.epsilon)
                    total_reward = rews + intrinsic_reward * CURIOSITY_BETA

                    policy_alg.process_env_step(total_reward, dones, infos)

                    obs = next_obs
                    critic_obs = infos["observations"]["critic"] if ("observations" in infos and "critic" in infos["observations"]) else obs
                    obs_n        = runner.obs_normalizer(obs)
                    critic_obs_n = runner.critic_obs_normalizer(critic_obs)

                    freshly_done = (done_traj_buffer[t] & (first_done_seen == max_episode_len))
                    if freshly_done.any():
                        idxs = torch.nonzero(freshly_done, as_tuple=False).squeeze(-1)
                        first_done_seen[idxs] = t + 1
                # ==========================================================================

                # ================== UPDATES (enable grads; AMP for ICM & PPO) =============
                if policy_alg.storage.step >= runner.num_steps_per_env:
                    num_transitions = policy_alg.storage.step
                    if num_transitions > 1:
                        with torch.enable_grad():
                            all_states  = policy_alg.storage.observations.view(-1, input_dim)
                            all_act     = policy_alg.storage.actions.view(-1, output_dim)
                            end_idx = (num_transitions - 1) * policy_alg.storage.num_envs
                            batch_states = all_states[:end_idx]
                            start_idx_next, end_idx_next = policy_alg.storage.num_envs, num_transitions * policy_alg.storage.num_envs
                            batch_next_states = all_states[start_idx_next:end_idx_next]
                            batch_act = all_act[:end_idx]

                            norm_batch_states      = state_normalizer.normalize(batch_states)
                            norm_batch_act         = action_normalizer.normalize(torch.clamp(batch_act, -100, 100))
                            norm_batch_next_states = state_normalizer.normalize(batch_next_states)

                            # AMP-scaled ICM update
                            with autocast(enabled=torch.cuda.is_available()):
                                f_loss, i_loss = icm(norm_batch_states, norm_batch_act, norm_batch_next_states)
                                icm_loss = 0.2 * f_loss.mean() + 0.8 * i_loss.mean()
                            icm_optimizer.zero_grad(set_to_none=True)
                            scaler.scale(icm_loss).backward()
                            scaler.step(icm_optimizer)
                            scaler.update()

                    with torch.enable_grad():
                        # Autocast PPO update (no external scaler since optimizer lives inside lib)
                        policy_alg.compute_returns(critic_obs_n)
                        mean_value_loss, mean_surrogate_loss, _, _, _ = policy_alg.update()
                # ==========================================================================

            action_mean = torch.mean(act).item()
            action_std  = torch.std(act).item()

            # ---- Build dedup embeddings (JOINTS ONLY, NO Z-SCORE) ----
            dof_pos_np  = dof_pos_traj_buffer.cpu().numpy()
            dof_vel_np  = dof_vel_traj_buffer.cpu().numpy()

            base_pos_np = base_pos_buffer.cpu().numpy()
            base_vel_np = base_vel_buffer.cpu().numpy()
            base_ang_np = base_ang_buffer.cpu().numpy()

            ball_pos_np = ball_pos_buffer.cpu().numpy()
            ball_vel_np = ball_vel_buffer.cpu().numpy()
            ball_ang_np = ball_ang_buffer.cpu().numpy()

            obs_np   = obs_traj_buffer.cpu().numpy()
            act_np   = act_traj_buffer.cpu().numpy()

            valid_len_np = np.where(first_done_seen.cpu().numpy() == max_episode_len,
                                    max_episode_len,
                                    first_done_seen.cpu().numpy())

            # [E, T, ...]
            dof_pos_batch  = np.transpose(dof_pos_np, (1, 0, 2))
            dof_vel_batch  = np.transpose(dof_vel_np, (1, 0, 2))

            base_pos_batch = np.transpose(base_pos_np,(1, 0, 2))
            base_vel_batch = np.transpose(base_vel_np,(1, 0, 2))
            base_ang_batch = np.transpose(base_ang_np,(1, 0, 2))

            ball_pos_batch = np.transpose(ball_pos_np, (1, 0, 2))
            ball_vel_batch = np.transpose(ball_vel_np, (1, 0, 2))
            ball_ang_batch = np.transpose(ball_ang_np, (1, 0, 2))


            obs_batch      = np.transpose(obs_np,     (1, 0, 2))
            act_batch      = np.transpose(act_np,     (1, 0, 2))

            # zero invalid tail (matches written data)
            for e in range(NUM_ENVS):
                L = valid_len_np[e]
                if L < max_episode_len:
                    dof_pos_batch[e, L:] = 0.0
                    dof_vel_batch[e, L:] = 0.0

                    base_pos_batch[e,   L:] = 0.0
                    base_vel_batch[e, L:] = 0.0
                    base_ang_batch[e, L:] = 0.0
                    ball_pos_batch[e, L:] = 0.0   # NEW (ball)
                    ball_vel_batch[e, L:] = 0.0   # NEW (ball)
                    ball_ang_batch[e, L:] = 0.0   # NEW (ball)


                    obs_batch[e,   L:] = 0.0
                    act_batch[e,   L:] = 0.0

            # Downsampled blocks (these are NumPy already in your code)
            dof_pos_dsmp  = dof_pos_batch[:, ::DOWNSAMPLE_FACTOR, :]
            base_vel_dsmp = base_vel_batch[:, ::DOWNSAMPLE_FACTOR, :]
            base_ang_dsmp = base_ang_batch[:, ::DOWNSAMPLE_FACTOR, :]

            # Flatten per episode
            J = dof_pos_dsmp.reshape(NUM_ENVS, -1)
            V = base_vel_dsmp.reshape(NUM_ENVS, -1)
            W = base_ang_dsmp.reshape(NUM_ENVS, -1)

            faiss.normalize_L2(J)
            faiss.normalize_L2(V)
            faiss.normalize_L2(W)

            new_embeddings = np.concatenate([J, V, W],axis=1)
            faiss.normalize_L2(new_embeddings)  # IMPORTANT: normalize after concat

            if faiss_index.ntotal == 0:
                unique_indices = np.arange(len(new_embeddings))
            else:
                D, I = faiss_index.search(x=new_embeddings, k=1)
                is_unique_mask = D[:, 0] < SIMILARITY_THRESHOLD
                unique_indices = np.where(is_unique_mask)[0]

            if unique_indices.size > 0:
                dof_pos_save_buffer.append(dof_pos_batch[unique_indices])
                dof_vel_save_buffer.append(dof_vel_batch[unique_indices])

                base_pos_save_buffer.append(base_pos_batch[unique_indices])
                base_vel_save_buffer.append(base_vel_batch[unique_indices])
                base_ang_save_buffer.append(base_ang_batch[unique_indices])

                ball_pos_save_buffer.append(ball_pos_batch[unique_indices])   # NEW (ball)
                ball_vel_save_buffer.append(ball_vel_batch[unique_indices])   # NEW (ball)
                ball_ang_save_buffer.append(ball_ang_batch[unique_indices])   # NEW (ball)


                obs_save_buffer.append(obs_batch[unique_indices])
                act_save_buffer.append(act_batch[unique_indices])

                valid_len_save_buffer.append(valid_len_np[unique_indices])

                faiss_index.add(x=new_embeddings[unique_indices])
                episodes_since_last_save += len(unique_indices)

            # Write in blocks
            if episodes_since_last_save >= WRITE_BUFFER_SIZE or total_episodes_saved + episodes_since_last_save >= EPISODES_TO_COLLECT:
                if dof_pos_save_buffer:
                    dof_pos_block  = np.concatenate(dof_pos_save_buffer,  axis=0)
                    dof_vel_block  = np.concatenate(dof_vel_save_buffer,  axis=0)

                    base_pos_block = np.concatenate(base_pos_save_buffer, axis=0)
                    base_vel_block = np.concatenate(base_vel_save_buffer, axis=0)
                    base_ang_block = np.concatenate(base_ang_save_buffer, axis=0)

                    ball_pos_block = np.concatenate(ball_pos_save_buffer, axis=0)  # NEW (ball)
                    ball_vel_block = np.concatenate(ball_vel_save_buffer, axis=0)  # NEW (ball)
                    ball_ang_block = np.concatenate(ball_ang_save_buffer, axis=0)  # NEW (ball)


                    obs_block  = np.concatenate(obs_save_buffer,      axis=0)
                    act_block  = np.concatenate(act_save_buffer,      axis=0)

                    vlen_block = np.concatenate(valid_len_save_buffer,axis=0)

                    num_in_block = dof_pos_block.shape[0]
                    start_idx = total_episodes_saved
                    end_idx = min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
                    num_to_write = end_idx - start_idx
                    if num_to_write > 0:
                        print(f"  ...Writing {num_to_write} episodes to disk...")
                        dof_pos_ds[start_idx:end_idx]   = dof_pos_block[:num_to_write]
                        dof_vel_ds[start_idx:end_idx]   = dof_vel_block[:num_to_write]

                        base_pos_ds[start_idx:end_idx]  = base_pos_block[:num_to_write]
                        base_vel_ds[start_idx:end_idx]  = base_vel_block[:num_to_write]
                        base_ang_ds[start_idx:end_idx]  = base_ang_block[:num_to_write]
                        ball_pos_ds[start_idx:end_idx] = ball_pos_block[:num_to_write]  # NEW (ball)
                        ball_vel_ds[start_idx:end_idx] = ball_vel_block[:num_to_write]  # NEW (ball)
                        ball_ang_ds[start_idx:end_idx] = ball_ang_block[:num_to_write]  # NEW (ball)


                        obs_ds[start_idx:end_idx]       = obs_block[:num_to_write]
                        act_ds[start_idx:end_idx]       = act_block[:num_to_write]

                        valid_len_ds[start_idx:end_idx] = vlen_block[:num_to_write]
                        total_episodes_saved = end_idx

                    dof_pos_save_buffer, dof_vel_save_buffer, = [], []
                    base_pos_save_buffer, base_vel_save_buffer, base_ang_save_buffer = [], [], []
                    ball_pos_save_buffer, ball_vel_save_buffer, ball_ang_save_buffer = [], [], []  # NEW (ball)

                    obs_save_buffer, act_save_buffer = [], []
                    valid_len_save_buffer = []
                    episodes_since_last_save = 0

            print(f"  ...Collected: {total_episodes_saved}/{EPISODES_TO_COLLECT} | PPO Loss: {mean_surrogate_loss:.3f} "
                  f"| Fwd Loss: {f_loss.mean().item():.3f} | Inv Loss: {i_loss.mean().item():.3f} "
                  f"| Action Mean: {action_mean:.3f} | Action Std: {action_std:.3f}")

        # Final flush if remaining
        if dof_pos_save_buffer:
            dof_pos_block = np.concatenate(dof_pos_save_buffer, axis=0)
            dof_vel_block   = np.concatenate(dof_vel_save_buffer,   axis=0)

            base_pos_block   = np.concatenate(base_pos_save_buffer,   axis=0)
            base_vel_block   = np.concatenate(base_vel_save_buffer,   axis=0)
            base_ang_block   = np.concatenate(base_ang_save_buffer,   axis=0)
            ball_pos_block = np.concatenate(ball_pos_save_buffer, axis=0)  # NEW (ball)
            ball_vel_block = np.concatenate(ball_vel_save_buffer, axis=0)  # NEW (ball)
            ball_ang_block = np.concatenate(ball_ang_save_buffer, axis=0)  # NEW (ball)

            obs_block   = np.concatenate(obs_save_buffer,   axis=0)
            act_block   = np.concatenate(act_save_buffer,   axis=0)

            vlen_block  = np.concatenate(valid_len_save_buffer, axis=0)

            num_in_block = dof_pos_block.shape[0]
            start_idx = total_episodes_saved
            end_idx = min(total_episodes_saved + num_in_block, EPISODES_TO_COLLECT)
            num_to_write = end_idx - start_idx
            if num_to_write > 0:
                print(f"  ...Writing final {num_to_write} episodes to disk...")
                dof_pos_ds[start_idx:end_idx] = dof_pos_block[:num_to_write]
                dof_vel_ds[start_idx:end_idx] = dof_vel_block[:num_to_write]

                base_pos_ds[start_idx:end_idx] = base_pos_block[:num_to_write]
                base_vel_ds[start_idx:end_idx] = base_vel_block[:num_to_write]
                base_ang_ds[start_idx:end_idx] = base_ang_block[:num_to_write]
                ball_pos_ds[start_idx:end_idx] = ball_pos_block[:num_to_write]  # NEW (ball)
                ball_vel_ds[start_idx:end_idx] = ball_vel_block[:num_to_write]  # NEW (ball)
                ball_ang_ds[start_idx:end_idx] = ball_ang_block[:num_to_write]  # NEW (ball)

                obs_ds[start_idx:end_idx]     = obs_block[:num_to_write]
                act_ds[start_idx:end_idx] = act_block[:num_to_write]

                valid_len_ds[start_idx:end_idx] = vlen_block[:num_to_write]
                total_episodes_saved = end_idx

    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Normalizers saved to '{norm_state_path}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")
