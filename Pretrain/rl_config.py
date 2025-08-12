from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml
import copy

@dataclass
class RLConfigs:
    env_cfg: Dict[str, Any]
    obs_cfg: Dict[str, Any]
    reward_cfg: Dict[str, Any]
    command_cfg: Dict[str, Any]
    train_cfg: Dict[str, Any]

def _safe_get(d: Dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_configs(yaml_path: str, exp_name: str, max_iterations: int) -> RLConfigs:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    rl = raw.get("rl", {})
    env_cfg = copy.deepcopy(rl.get("env", {}))
    obs_cfg = copy.deepcopy(rl.get("obs", {}))
    reward_cfg = copy.deepcopy(rl.get("reward", {}))
    command_cfg = copy.deepcopy(rl.get("command", {}))
    train_cfg = copy.deepcopy(rl.get("train", {}))

    # Ensure required keys exist
    obs_cfg.setdefault("num_obs", None)
    obs_cfg.setdefault("obs_scales", {"lin_vel":2.0, "ang_vel":0.25, "dof_pos":1.0, "dof_vel":0.05})

    # Runner defaults
    train_cfg.setdefault("runner", {})
    train_cfg["runner"].setdefault("experiment_name", exp_name)
    train_cfg["runner"].setdefault("max_iterations", max_iterations)

    return RLConfigs(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)