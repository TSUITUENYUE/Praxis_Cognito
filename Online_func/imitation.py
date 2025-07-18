import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from codebook import Codebook
from Pretrain.dataset import TrajectoryDataset
from Pretrain.utils import *
from torch.utils.data import DataLoader
import genesis as gs


class ImitationModule:
    def __init__(self, model, cfg: DictConfig):
        self.vae = model
        self.agent = self.vae.agent
        self.config = cfg
        self.device = 'cuda'


    def imitate(self, demo, *codebook: Codebook):
        dataset = TrajectoryDataset(
            processed_path=self.config.processed_path,
            source_path=demo,
            agent=self.agent)

        dataloader = DataLoader(dataset,
                                batch_size=self.config.batch_size,
                                num_workers=4)

        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.device)
        loss = []
        cmd = []
        for i, (graph_x, orig_traj) in enumerate(dataloader):
            output = self.vae(graph_x, edge_index)
            recon_traj = output[0]
            joint_cmd = output[1]
            z = output[2]
            codebook.update(z)
            recon_loss = F.mse_loss(recon_traj, orig_traj)
            loss.append(recon_loss)
            cmd.append(joint_cmd)

        return loss, cmd

    def visualize_in_sim(self, cmd):
        gs.init(theme="light", logging_level='warning')

        NUM_ENVS = 1
        scene = gs.Scene(
            show_viewer= True,
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.Rasterizer(),
        )





