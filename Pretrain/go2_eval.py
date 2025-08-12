import argparse
import os
import pickle
from importlib import metadata

import torch
import numpy as np
import time
import h5py
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pretrain.rl_config import load_configs
from Model.agent import Agent

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


def _load_dataset_cfg(yaml_path: str):
    """
    Reads dataset timing from the unified config. Supported keys:
      dataset.max_episode_seconds
      dataset.frame_rate
    Or nested:
      dataset.collection.seconds
      dataset.collection.fps | dataset.collection.frame_rate
    """
    out = {}
    with open(yaml_path, "r") as f:
        root = yaml.safe_load(f) or {}


    ds = root.get("dataset", {})

    # flat style
    out["seconds"] = float(ds["max_episode_seconds"])
    out["fps"] = int(ds["frame_rate"])


    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=199)
    parser.add_argument("--config", type=str, default="./conf/go2.yaml")  # unified config
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    gs.init()

    # Load RL configs + Agent from unified config
    cfgs = load_configs(args.config, args.exp_name, max_iterations=1)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = (
        cfgs.env_cfg, cfgs.obs_cfg, cfgs.reward_cfg, cfgs.command_cfg, cfgs.train_cfg
    )


    agent = Agent(**args.config.agent)

    MAX_EPISODE_SECONDS = args.config.dataset.max_episode_seconds
    FRAME_RATE = args.config.dataset.frame_rate

    log_dir = (
        f"Pretrain/primitives/{args.exp_name}"
    )

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        agent=agent,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    joint_names = agent.joint_name
    dof_indices = np.array([env.robot.get_joint(name).dof_idx_local for name in joint_names])
    n_dofs = len(dof_indices)

    # --- Data generation parameters ---
    NUM_ENVS = 1
    EPISODES_TO_COLLECT = int(args.episodes)
    AGENT = agent.name if hasattr(agent, "name") else "go2"
    out_dir = f"./data/{AGENT}/demo"
    os.makedirs(out_dir, exist_ok=True)
    SAVE_FILENAME = f"{out_dir}/walk.h5"

    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)

    # Pre-allocate trajectory buffers on the GPU
    agent_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs), device='cuda')
    obj_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, 3), device='cuda')  # placeholder

    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)

    start_time = time.time()
    with h5py.File(SAVE_FILENAME, 'w') as f:
        agent_ds = f.create_dataset('agent_trajectories',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs),
                                    dtype='float32', compression="gzip")
        obj_ds   = f.create_dataset('obj_trajectories',
                                    shape=(EPISODES_TO_COLLECT, max_episode_len, 3),
                                    dtype='float32', compression="gzip")
        with torch.no_grad():
            episodes_saved = 0
            while episodes_saved < EPISODES_TO_COLLECT:
                obs, _ = env.reset()
                for t in range(max_episode_len):
                    agent_traj_buffer[t] = env.robot.get_dofs_position(dof_indices).to('cuda')
                    actions = policy(obs)
                    obs, rews, dones, infos = env.step(actions)
                    # If you later add an object, fill obj_traj_buffer[t] here
                agent_ds[episodes_saved:episodes_saved+NUM_ENVS] = agent_traj_buffer.cpu().numpy().transpose(1,0,2)
                obj_ds[episodes_saved:episodes_saved+NUM_ENVS]   = obj_traj_buffer.cpu().numpy().transpose(1,0,2)
                episodes_saved += NUM_ENVS

    print(f"Saved dataset to {SAVE_FILENAME}")
    print(f"Used timing from config: seconds={MAX_EPISODE_SECONDS}, fps={FRAME_RATE}")


if __name__ == "__main__":
    main()

"""
# evaluation with data collection
python go2_eval.py --config ./conf/go2v.yaml -e go2-walking --ckpt 100 --episodes 1
"""
