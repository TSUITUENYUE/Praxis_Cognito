import argparse
import os
import pickle
import shutil
import hydra
from omegaconf import DictConfig, OmegaConf
from importlib import metadata
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env_primitives import Go2Env
from Model.agent import Agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="1go2base")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=401)
    parser.add_argument("--config", type=str, default="../conf/go2base.yaml")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # Create log dir
    log_dir = f"Pretrain/primitives/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Load configs from YAML (with sensible fallbacks)
    config_dir = os.path.dirname(args.config) or "."
    config_name = os.path.basename(args.config).rstrip('.yaml')
    hydra.initialize(config_path=config_dir)
    config = hydra.compose(config_name)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    rl_config = config.rl
    MAX_EPISODE_SECONDS = config.dataset.max_episode_seconds
    FRAME_RATE = config.dataset.frame_rate

    # ---- Agent & Env ----
    agent = Agent(**config.agent).to('cuda')
    NUM_ENVS = args.num_envs
    env = Go2Env(
        num_envs=NUM_ENVS,
        env_cfg=rl_config.env,
        obs_cfg=rl_config.obs,
        reward_cfg=rl_config.reward,
        command_cfg=rl_config.command,
        show_viewer=False,
        agent=agent,
    )
    train_cfg = OmegaConf.to_container(rl_config.train, resolve=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

"""
# training
python go2_train.py --config go2.yaml -e go2-walking -B 2048 --max_iterations 200
"""
