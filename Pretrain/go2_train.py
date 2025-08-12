import argparse
import os
import pickle
import shutil
from importlib import metadata
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from Model.agent import Agent
from rl_config import load_configs, load_agent_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--config", type=str, default="./conf/go2.yaml")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # Load configs from YAML (with sensible fallbacks)
    cfgs = load_configs(args.config, args.exp_name, args.max_iterations)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs.env_cfg, cfgs.obs_cfg, cfgs.reward_cfg, cfgs.command_cfg, cfgs.train_cfg
    agent_cfg = load_agent_cfg(args.config)
    agent = Agent(
        name=agent_cfg.get('name', 'go2'),
        urdf=agent_cfg['urdf'],
        n_dofs=agent_cfg['n_dofs'],
        object_dim=agent_cfg.get('object_dim', 3),
        joint_name=agent_cfg['joint_name'],
        end_effector=agent_cfg.get('end_effector', []),
        init_angles=agent_cfg.get('init_angles', [0.0]*agent_cfg['n_dofs']),
    )


    # Create log dir
    log_dir = f"./primitives/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save configs for reproducibility (both yaml copy and legacy pkl expected by the old eval)
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))
    # also copy the yaml
    try:
        import shutil as _sh
        _sh.copy2(args.config, os.path.join(log_dir, "config.yaml"))
    except Exception:
        pass

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        agent=agent,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

"""
# training
python go2_train.py --config go2.yaml -e go2-walking -B 2048 --max_iterations 200
"""
