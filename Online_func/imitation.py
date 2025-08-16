# imitation.py (clean, new-contract only)

from __future__ import annotations
import os, h5py, math
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import List

import genesis as gs
from rsl_rl.modules import EmpiricalNormalization

from Pretrain.dataset import TrajectoryDataset
from Pretrain.utils import build_edge_index


class ImitationModule:
    def __init__(self, model, cfg: DictConfig):
        self.vae = model
        self.agent = self.vae.agent
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device).eval()

    # ----------------- helpers -----------------
    @staticmethod
    def _denorm_positions(traj_norm: np.ndarray, pos_mean: np.ndarray, pos_std: np.ndarray) -> np.ndarray:
        T, D = traj_norm.shape
        x = traj_norm.reshape(T, D // 3, 3)
        x = x * pos_std[None, None, :] + pos_mean[None, None, :]
        return x.reshape(T, D)

    @staticmethod
    def _load_obs_normalizer_from_h5(h5_path: str, obs_dim: int, device: torch.device) -> EmpiricalNormalization:
        with h5py.File(h5_path, "r") as f:
            if "meta" not in f or "normalizer_pt" not in f["meta"].attrs:
                raise FileNotFoundError("Dataset missing meta.normalizer_pt attribute.")
            norm_path = f["meta"].attrs["normalizer_pt"]
            if isinstance(norm_path, bytes):
                norm_path = norm_path.decode("utf-8")
        if not os.path.isfile(norm_path):
            raise FileNotFoundError(f"Normalizer state not found: {norm_path}")

        state = torch.load(norm_path, map_location="cpu")
        norm = EmpiricalNormalization(obs_dim)
        norm.load_state_dict(state["obs_norm"])
        norm.to(device).eval()
        for p in norm.parameters():  # freeze
            p.requires_grad_(False)
        return norm

    # ----------------- evaluation -----------------
    def imitate(self, demo_h5_path: str):
        """
        Returns:
          losses: list of float (MSE on normalized FK/object space, same as training)
          joint_cmds: list of [B,T,DoF] tensors from model rollout (decoder + surrogate)
        """
        dataset = TrajectoryDataset(
            processed_path=self.cfg.processed_path,
            source_path=demo_h5_path,
            agent=self.agent,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        obs_dim = getattr(dataset, "obs_dim", self.cfg.obs_dim)
        normalizer = self._load_obs_normalizer_from_h5(demo_h5_path, obs_dim, self.device)

        edge_index = build_edge_index(self.agent.fk_model.to(self.device),
                                      self.agent.end_effector, self.device)

        losses, joint_cmds = [], []
        self.vae.eval()
        with torch.no_grad():
            for (graph_x, obs, act, q, dq, mask) in loader:
                graph_x = graph_x.to(self.device, non_blocking=True)
                obs     = obs.to(self.device, non_blocking=True)
                q       = q.to(self.device, non_blocking=True)
                dq      = dq.to(self.device, non_blocking=True)
                mask    = mask.to(self.device, non_blocking=True)[:, :, None]

                # forward with tf_ratio=0 for pure rollout
                out = self.vae(
                    graph_x, edge_index, mask, normalizer,
                    obs_seq=obs, q=q, dq=dq, tf_ratio=0.0,
                )
                # new tuple: (recon_mu, joints_seq, actions_seq, pre_mu_seq, log_std_seq, logsig_seq, …)
                recon_mu, joints_seq = out[0], out[1]

                loss = F.mse_loss(recon_mu, graph_x, reduction="mean").item()
                losses.append(loss)
                joint_cmds.append(joints_seq.detach().cpu())

        return losses, joint_cmds

    # ----------------- visualization -----------------
    def visualize_in_sim(self, demo_h5_path: str, index: int = 0, save_path: str = "animation.mp4"):
        dataset = TrajectoryDataset(processed_path="./Pretrain/data/go2/16384 250000 2 30/preprocess.h5",
                                    source_path="./Pretrain/data/go2/16384 250000 2 30/16384 250000 2 30.h5",
                                    agent=self.agent)
        pos_mean = dataset.pos_mean.cpu().numpy()
        pos_std  = dataset.pos_std.cpu().numpy()
        obs_dim  = 51
        dataset = TrajectoryDataset(processed_path=self.cfg.processed_path,
                                    source_path=demo_h5_path,
                                    agent=self.agent)
        #normalizer = self._load_obs_normalizer_from_h5(demo_h5_path, obs_dim, self.device)
        normalizer = torch.nn.Identity().to(self.device)
        graph_x_np, obs_np, act_np, q_np, dq_np, mask_np = dataset[index]
        graph_x = torch.tensor(graph_x_np, device=self.device).unsqueeze(0)
        obs     = torch.tensor(obs_np,     device=self.device).unsqueeze(0)
        q       = torch.tensor(q_np,       device=self.device).unsqueeze(0)
        dq      = torch.tensor(dq_np,      device=self.device).unsqueeze(0)
        mask    = torch.tensor(mask_np,    device=self.device).unsqueeze(0)[:, :, None]

        edge_index = build_edge_index(self.agent.fk_model.to(self.device),
                                      self.agent.end_effector, self.device)

        self.vae.eval()
        with torch.no_grad():
            out = self.vae(
                graph_x, edge_index, mask, normalizer,
                obs_seq=obs, q=q, dq=dq, tf_ratio=0.0,
            )
            recon_mu, joints_seq = out[0], out[1]

        # Optional metric (world space)
        recon_world = self._denorm_positions(recon_mu.squeeze(0).cpu().numpy(), pos_mean, pos_std)
        target_world= self._denorm_positions(graph_x.view(recon_mu.shape).squeeze(0).cpu().numpy(),  pos_mean, pos_std)
        print(f"[Diag] FK-space MSE (world units): {((recon_world - target_world)**2).mean():.6f}")

        # -------- Genesis playback of predicted joints --------
        gs.init(theme="light", logging_level="warning")
        scene = gs.Scene(
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(res=(1280, 960), camera_pos=(0.0, 3.5, 2.5),
                                                    camera_lookat=(0.0, 0.0, 0.5), camera_fov=40, max_FPS=30),
            vis_options=gs.options.VisOptions(show_world_frame=True, world_frame_size=1.0,
                                              plane_reflection=True, ambient_light=(0.1, 0.1, 0.1)),
            renderer=gs.renderers.Rasterizer(),
        )
        _ = scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.URDF(file=self.agent.urdf, collision=True, fixed=False))
        cam = scene.add_camera(res=(1280, 960), pos=(0.0, 3.5, 2.5), lookat=(0.0, 0.0, 0.5), fov=40, GUI=False)
        scene.build()
        robot.set_pos(np.array([0.0, 0.0, 0.42]))
        robot.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        dof_indices = np.array([robot.get_joint(name).dof_idx_local for name in self.agent.joint_name], dtype=np.int32)
        seq = joints_seq.squeeze(0).cpu().numpy()  # [T, DoF]

        cam.start_recording()
        for t in range(seq.shape[0]):
            robot.set_dofs_position(seq[t], dofs_idx_local=dof_indices)
            scene.step()
            cam.render()
        cam.stop_recording(save_to_filename=save_path, fps=30)
        print(f"Saved sim video to: {save_path}")
