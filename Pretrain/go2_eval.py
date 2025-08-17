# go2_eval_new.py  — aligned with new dataset criteria

import argparse
import os
import pickle
from importlib import metadata
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import time
import h5py
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import genesis as gs

# local imports
from Model.agent import Agent
from go2_env import Go2Env

# --- rsl-rl import guard (match your constraint) ---
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../conf/go2.yaml", help="unified YAML config, e.g. ./conf/go2.yaml")
    parser.add_argument("-e", "--exp_name", default="go2-walking",type=str,help="experiment name under Pretrain/primitives/")
    parser.add_argument("--ckpt", type=int, default=199, help="checkpoint index to load")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to record")
    parser.add_argument("--viewer", action="store_true", help="show viewer while recording")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # ---- Load configs from unified YAML ----
    config_dir = os.path.dirname(args.config) or "."
    config_name = os.path.basename(args.config).rstrip('.yaml')
    hydra.initialize(config_path=config_dir)
    config = hydra.compose(config_name)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    rl_config = config.rl
    MAX_EPISODE_SECONDS = config.dataset.max_episode_seconds
    FRAME_RATE = config.dataset.frame_rate

    # ---- Agent & Env ----
    agent = Agent(**config.agent)
    NUM_ENVS = 1
    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=rl_config.env,
        obs_cfg=rl_config.obs,
        reward_cfg=rl_config.reward,
        command_cfg=rl_config.command,
        show_viewer=True,
        agent=agent,
    )

    # ---- PPO runner + checkpoint ----
    log_dir = f"Pretrain/primitives/{args.exp_name}"

    train_cfg = OmegaConf.to_container(rl_config.train, resolve=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy_alg = runner.alg  # use the same act(obs_n, critic_obs_n) path as in generator

    # ---- Output dataset path (match generator naming) ----
    EPISODES = int(args.episodes)
    AGENT = getattr(agent, "name", "go2")
    base_dir = f"./Pretrain/data/{AGENT}/demo"
    os.makedirs(base_dir, exist_ok=True)
    SAVE_FILENAME = f"{base_dir}/walk.h5"

    # Save runner normalizers so training can reuse them
    norm_state_path = os.path.join(base_dir, "normalizers.pt")
    torch.save(
        {
            "obs_norm": runner.obs_normalizer.state_dict(),
            "critic_obs_norm": runner.critic_obs_normalizer.state_dict(),
        },
        norm_state_path,
    )

    # ---- Buffers ----
    joint_names = agent.joint_name
    dof_indices = np.array([env.robot.get_joint(name).dof_idx_local for name in joint_names], dtype=np.int32)
    n_dofs = len(dof_indices)
    max_T = int(MAX_EPISODE_SECONDS * FRAME_RATE)

    # GPU buffers
    agent_traj_buffer = torch.zeros((max_T, NUM_ENVS, n_dofs), device=gs.device, dtype=torch.float32)
    obj_traj_buffer   = torch.zeros((max_T, NUM_ENVS, 3),     device=gs.device, dtype=torch.float32)
    obs_traj_buffer   = torch.zeros((max_T, NUM_ENVS, env.num_obs),    device=gs.device, dtype=torch.float32)
    act_traj_buffer   = torch.zeros((max_T, NUM_ENVS, env.num_actions), device=gs.device, dtype=torch.float32)
    vel_traj_buffer   = torch.zeros((max_T, NUM_ENVS, n_dofs), device=gs.device, dtype=torch.float32)
    done_traj_buffer  = torch.zeros((max_T, NUM_ENVS), device=gs.device, dtype=torch.bool)

    # ---- Collect ----
    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)

    start = time.time()
    with h5py.File(SAVE_FILENAME, "w") as f:
        agent_ds   = f.create_dataset("agent_trajectories", (EPISODES, max_T, n_dofs), dtype="float32", compression="gzip")
        obj_ds     = f.create_dataset("obj_trajectories",   (EPISODES, max_T, 3),      dtype="float32", compression="gzip")
        obs_ds     = f.create_dataset("obs",                (EPISODES, max_T, env.num_obs), dtype="float32", compression="gzip")
        actions_ds = f.create_dataset("actions",            (EPISODES, max_T, env.num_actions), dtype="float32", compression="gzip")
        vel_ds     = f.create_dataset("dof_vel",            (EPISODES, max_T, n_dofs), dtype="float32", compression="gzip")
        vlen_ds    = f.create_dataset("valid_length",       (EPISODES,), dtype="int32", compression="gzip")

        # meta (match generator)
        meta = f.create_group("meta")
        meta.attrs["dt"] = float(env.dt)
        act_scale = env.env_cfg.get("action_scale", 0.25)
        if isinstance(act_scale, (list, tuple, np.ndarray, torch.Tensor)):
            meta.create_dataset("action_scale", data=np.asarray(act_scale, dtype=np.float32))
        else:
            meta.attrs["action_scale"] = float(act_scale)
        meta.create_dataset("default_dof_pos", data=np.asarray(env.default_dof_pos.cpu(), dtype=np.float32))
        meta.attrs["normalizer_pt"] = os.path.abspath(norm_state_path)

        episodes_saved = 0
        with torch.no_grad():
            while episodes_saved < EPISODES:
                # reset and normalized actor/critic obs (exactly as in generator)
                obs, extras = env.reset()
                critic_obs = extras["observations"].get("critic", obs)
                obs_n        = runner.obs_normalizer(obs)
                critic_obs_n = runner.critic_obs_normalizer(critic_obs)

                first_done_idx = max_T  # default: no done
                for t in range(max_T):
                    # store raw obs before stepping
                    obs_traj_buffer[t] = obs

                    # normalized acting (same path as training/generation)
                    actions = policy_alg.act(obs_n, critic_obs_n)
                    act_traj_buffer[t] = actions

                    # step sim
                    next_obs, rews, dones, infos = env.step(actions)
                    done_traj_buffer[t] = (dones.squeeze(-1) > 0) if dones.ndim == 2 else (dones > 0)

                    # record kinematics AFTER action (post-step)
                    q_now  = env.robot.get_dofs_position(env.motors_dof_idx)
                    dq_now = env.robot.get_dofs_velocity(env.motors_dof_idx)
                    vel_traj_buffer[t]   = dq_now
                    agent_traj_buffer[t] = q_now

                    # optional object (keep zero if no object present)
                    if hasattr(env, "ball"):
                        obj_traj_buffer[t] = env.ball.get_pos()

                    # advance obs & re-compute normalized obs
                    obs = next_obs
                    critic_obs = infos["observations"]["critic"] if ("observations" in infos and "critic" in infos["observations"]) else obs
                    obs_n        = runner.obs_normalizer(obs)
                    critic_obs_n = runner.critic_obs_normalizer(critic_obs)

                    # first done index
                    if (first_done_idx == max_T) and done_traj_buffer[t, 0].item():
                        first_done_idx = t + 1  # inclusive length

                # write episode (masking frames after first done with zeros to match generator semantics)
                T_valid = first_done_idx
                # move to CPU & transpose to [E, T, ...] -> here E=1, so squeeze later
                agent_np = agent_traj_buffer.cpu().numpy().transpose(1, 0, 2)  # [1, T, dof]
                obj_np   = obj_traj_buffer.cpu().numpy().transpose(1, 0, 2)
                obs_np   = obs_traj_buffer.cpu().numpy().transpose(1, 0, 2)
                act_np   = act_traj_buffer.cpu().numpy().transpose(1, 0, 2)
                vel_np   = vel_traj_buffer.cpu().numpy().transpose(1, 0, 2)

                # zero out invalid tail
                if T_valid < max_T:
                    agent_np[0, T_valid:] = 0.0
                    obj_np[0,   T_valid:] = 0.0
                    obs_np[0,   T_valid:] = 0.0
                    act_np[0,   T_valid:] = 0.0
                    vel_np[0,   T_valid:] = 0.0

                ep = episodes_saved
                agent_ds[ep:ep+1]   = agent_np
                obj_ds[ep:ep+1]     = obj_np
                obs_ds[ep:ep+1]     = obs_np
                actions_ds[ep:ep+1] = act_np
                vel_ds[ep:ep+1]     = vel_np
                vlen_ds[ep]         = T_valid

                episodes_saved += 1
                print(f"[Episode {ep+1}/{EPISODES}] valid_len={T_valid}/{max_T}")

    dur = time.time() - start
    print("\nData collection complete.")
    print(f"Saved to: {SAVE_FILENAME}")
    print(f"Normalizers saved to: {norm_state_path}")
    print(f"seconds={MAX_EPISODE_SECONDS}, fps={FRAME_RATE}, wall-clock={dur:.2f}s")


if __name__ == "__main__":
    main()
