import hydra
from omegaconf import DictConfig, OmegaConf
from Model.vae import IntentionVAE
from Online_func.imitation import ImitationModule
from Pretrain.train import *
from Model.agent import *
from codebook import Codebook
from Pretrain.generate_dataset import generate
from Pretrain.go2_env import Go2Env
import argparse
import os
import genesis as gs

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class Runner:
    def __init__(self, mode, config: DictConfig):
        self.mode = mode
        self.config = config
        self.agent = Agent(**config.agent)
        if self.mode != "generate":
            self.model = IntentionVAE(agent=self.agent, obs_dim=config.rl.obs.num_obs, fps=config.dataset.frame_rate, cfg=config.rl, **config.model.vae)

            if self.mode == "train":
                self.trainer = Trainer(self.model, config.rl, config.trainer)
            if self.mode != "train":
                self.codebook = Codebook(**config.codebook)
                checkpoint_path = self.config.trainer.save_path + f"vae_checkpoint_epoch_{self.config.trainer.num_epochs}.pth"
                state_dict = torch.load(checkpoint_path, map_location=self.config.trainer.device)
                self.model = torch.compile(self.model).to(self.config.trainer.device)
                self.model.load_state_dict(state_dict)
                self.imitator = ImitationModule(model=self.model, cfg = config.imitator)

    def run(self, demo=None):
        if self.mode == "generate":
            generate(self.config)
        elif self.mode == "train":
            self.trainer.train()
        elif self.mode == "imitate":
            #self.imitator.imitate(demo, self.codebook)
            self.imitator.visualize_in_sim(demo, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--config", default="./conf/go2.yaml")
    parser.add_argument("--demo", default=None)
    args, unknown_args = parser.parse_known_args()
    if args.mode == "imitate" and args.demo is None:
        parser.error("demo is required for imitate mode")

    config_dir = os.path.dirname(args.config) or "."
    config_name = os.path.basename(args.config).rstrip('.yaml')
    hydra.initialize(version_base=None, config_path=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=unknown_args)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

    runner = Runner(args.mode, cfg)
    runner.run(demo=args.demo)