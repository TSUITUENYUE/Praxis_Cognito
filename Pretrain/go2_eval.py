import argparse
import os
import pickle
from importlib import metadata

import torch
import numpy as np
import time
import h5py

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

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    # --- Configuration for Data Generation ---
    NUM_ENVS = 1 # Increase for more parallel data generation
    EPISODES_TO_COLLECT = 1  # The number of episodes to generate
    MAX_EPISODE_SECONDS = 5
    FRAME_RATE = 30  # Based on control frequency
    AGENT = "go2"
    path = f"./data/{AGENT}/demo"
    if not os.path.exists(path): os.makedirs(path)
    SAVE_FILENAME = f"{path}/walk.h5"

    gs.init()

    log_dir = f"./Pretrain/primitives/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"./Pretrain/primitives/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=NUM_ENVS < 128,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
    dof_indices = np.array([env.robot.get_joint(name).dof_idx_local for name in joint_names])
    print(dof_indices)
    n_dofs = len(dof_indices)

    # --- Main Continuous Simulation Loop ---
    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)
    total_episodes_saved = 0

    # Pre-allocate trajectory buffers on the GPU
    agent_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs), device='cuda')
    obj_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, 3), device='cuda')
    print(f"Starting data generation. Target: {EPISODES_TO_COLLECT} episodes.")
    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()

    # Open HDF5 file once with a 'with' statement to ensure it's properly closed
    with h5py.File(SAVE_FILENAME, 'w') as f:
        print(f"Opened HDF5 file '{SAVE_FILENAME}' for writing.")

        # Create large datasets for all episodes (improvement for I/O speed)
        agent_ds = f.create_dataset('agent_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs), dtype='float32', compression="gzip")
        obj_ds = f.create_dataset('obj_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                    dtype='float32', compression="gzip")
        with torch.no_grad():
            while total_episodes_saved < EPISODES_TO_COLLECT:
                obs, _ = env.reset()

                for t in range(max_episode_len):
                    # Store data directly into the GPU buffer at the current timestep
                    agent_traj_buffer[t] = env.robot.get_dofs_position(dof_indices).to('cuda')
                    #obj_traj_buffer[t] = torch.tensor([0,0,0],device='cuda')
                    # Policy and Simulation Step
                    actions = policy(obs)
                    obs, rews, dones, infos = env.step(actions)

                # --- BATCH PROCESSING AND SAVING ---
                # Transfer the entire batch of trajectories from GPU to CPU
                agent_data_batch_np = agent_traj_buffer.cpu().numpy()
                obj_data_batch_up = obj_traj_buffer.cpu().numpy()
                # Write batch as slices to the large datasets
                start_idx = total_episodes_saved
                end_idx = min(start_idx + NUM_ENVS, EPISODES_TO_COLLECT)
                num_this_batch = end_idx - start_idx

                agent_ds[start_idx:end_idx] = np.transpose(agent_data_batch_np[:, :num_this_batch, :], (1, 0, 2))
                obj_ds[start_idx:end_idx] = np.transpose(obj_data_batch_up[:, :num_this_batch, :], (1, 0, 2))
                total_episodes_saved += num_this_batch

                print(f"  ...Collected and saved episodes up to {total_episodes_saved}/{EPISODES_TO_COLLECT}")

    # --- Finalize ---
    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")



if __name__ == "__main__":
    main()

"""
# evaluation with data collection
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""