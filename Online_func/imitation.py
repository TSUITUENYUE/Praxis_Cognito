import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from omegaconf import DictConfig
from codebook import Codebook
from Pretrain.dataset import TrajectoryDataset
from Pretrain.utils import build_edge_index
from torch.utils.data import DataLoader
import genesis as gs
import numpy as np


class ImitationModule:
    def __init__(self, model, cfg: DictConfig):
        self.vae = model
        self.agent = self.vae.agent
        self.config = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device).eval()

    # ---------- helpers ----------
    @staticmethod
    def _denorm_positions(traj_norm: np.ndarray,
                          pos_mean: np.ndarray,
                          pos_std: np.ndarray) -> np.ndarray:
        """
        Denormalize a flattened [T, (num_links+1)*3] trajectory (agent links + object).
        """
        T, D = traj_norm.shape
        num_nodes = D // 3  # (num_links + 1)
        x = traj_norm.reshape(T, num_nodes, 3)
        x = x * pos_std[None, None, :] + pos_mean[None, None, :]
        return x.reshape(T, D)

    @staticmethod
    def _extract_indices_for_robot(urdf_robot, joint_names: List[str]) -> np.ndarray:
        return np.array([urdf_robot.get_joint(name).dof_idx_local for name in joint_names], dtype=np.int32)

    @staticmethod
    def _extract_ee_positions(flat_agent_pos: np.ndarray, ee_indices: List[int]) -> np.ndarray:
        """
        flat_agent_pos: [T, num_links*3] (denormalized)
        returns EE positions concatenated: [T, len(ee_indices)*3]
        """
        T, D = flat_agent_pos.shape
        num_links = D // 3
        out = np.zeros((T, len(ee_indices) * 3), dtype=np.float32)
        for j, idx in enumerate(ee_indices):
            start = idx * 3
            out[:, j * 3:(j + 1) * 3] = flat_agent_pos[:, start:start + 3]
        return out

    def _unpack_forward(self, out_tuple):
        """
        Support both forward signatures:
        - (recon_mu, joint_cmd, z, ...)
        - (recon_mu, joint_cmd, log_sigma, z, ...)
        Returns: recon_mu, joint_cmd, z (z may be None if not present)
        """
        recon_mu = out_tuple[0]
        joint_cmd = out_tuple[1]
        z = None
        # try to find a latent tensor by last-dim match
        for t in out_tuple[2:]:
            if torch.is_tensor(t) and t.dim() >= 2 and t.shape[-1] == getattr(self.vae, "latent_dim", t.shape[-1]):
                z = t
                break
        return recon_mu, joint_cmd, z

    # ---------- main APIs ----------
    def imitate(self, demo_h5_path: str, codebook: Optional[Codebook] = None):
        """
        Run imitation on a dataset built from `demo_h5_path`.
        Returns: (loss_list, joint_cmd_list) where losses are floats and joint_cmd_list is a list of tensors [B,T,DoF].
        """
        dataset = TrajectoryDataset(
            processed_path=self.config.processed_path,
            source_path=demo_h5_path,
            agent=self.agent
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.device)

        losses: List[float] = []
        cmds: List[torch.Tensor] = []

        self.vae.eval()
        with torch.no_grad():
            for graph_x, orig_traj in loader:
                graph_x = graph_x.to(self.device, non_blocking=True)
                orig_traj = orig_traj.to(self.device, non_blocking=True)

                out = self.vae(graph_x, edge_index, teacher_joint=None)
                recon_mu, joint_cmd, z = self._unpack_forward(out)

                # recon_mu and orig_traj are in normalized position space
                recon_loss = F.mse_loss(recon_mu, orig_traj, reduction="mean").item()
                losses.append(recon_loss)
                cmds.append(joint_cmd.detach().cpu())

                if codebook is not None and z is not None:
                    codebook.update(z.detach().cpu())

        return losses, cmds

    def visualize_in_sim(self, demo, index: int = 0, save_path: str = "animation.mp4"):
        """
        Visualize the model rollout in Genesis for one sample from the processed dataset.
        Uses the model's joint_cmd to drive the robot; denormalizes positions for optional debugging/plots.
        """

        dataset = TrajectoryDataset(
            processed_path="./Pretrain/data/go2/128 128 5 30/preprocess.h5",
            agent=self.agent
        )
        pos_mean = dataset.pos_mean.cpu().numpy()  # [3]
        pos_std = dataset.pos_std.cpu().numpy()    # [3]
        print(pos_mean, pos_std)
        dataset = TrajectoryDataset(
            processed_path=self.config.processed_path,
            source_path=demo,
            agent=self.agent
        )


        # get one sample
        graph_x_np, orig_traj_np, joint_traj_np = dataset[index]  # both normalized

        # torch tensors
        graph_x = torch.tensor(graph_x_np, device=self.device).unsqueeze(0)  # [1,T,D]
        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.device)

        self.vae.eval()
        with torch.no_grad():
            out = self.vae(graph_x, edge_index, teacher_joint=None)
            # adapt to your forward signature
            recon_mu, joint_cmd, log_sigma = out[0], out[1], out[2]  # if you return log_sigma
            sigma = torch.exp(log_sigma)
            r = (torch.tensor(orig_traj_np, device=sigma.device) - recon_mu) / sigma

            print("NLL (mean):", (0.5 * ((r) ** 2) + torch.log(sigma) + 0.5 * np.log(2 * np.pi)).mean().item())
            print("E[r^2] (want ≈ 1):", r.pow(2).mean().item())
            print("sigma min/median/mean/max:",
                  sigma.min().item(), sigma.median().item(), sigma.mean().item(), sigma.max().item())
            frac_at_floor = (sigma <= (self.vae.decoder.sigma_min + 1e-8)).float().mean().item()
            print("fraction at floor:", frac_at_floor)


        # to numpy
        recon_mu_np = recon_mu.squeeze(0).cpu().numpy()  # normalized [T, pos_dim]
        joint_cmd_np = joint_cmd.squeeze(0).cpu().numpy()  # [T, DoF]
        orig_traj_np = np.asarray(orig_traj_np)  # normalized [T, pos_dim]

        # ---------- denormalize (for any inspection you want) ----------
        recon_pos_world = self._denorm_positions(recon_mu_np, pos_mean, pos_std)  # [T, pos_dim] in meters
        orig_pos_world  = self._denorm_positions(orig_traj_np, pos_mean, pos_std)


        # optional FK-space MSE in world units (sanity/debug)
        mse_world = ((recon_pos_world - orig_pos_world) ** 2).mean()
        print(f"FK-space MSE (world units): {mse_world:.6f}")

        # Split agent vs object (if needed for plotting)
        pos_dim = recon_pos_world.shape[-1]
        num_nodes = pos_dim // 3
        num_links = num_nodes - 1
        agent_dim = num_links * 3
        object_pos_world = recon_pos_world[:, agent_dim:]  # if you want to render an object

        # Extract EE positions (denormed) if you want to compare
        # ee_recon = self._extract_ee_positions(recon_pos_world[:, :agent_dim], self.agent.end_effector)
        # ee_orig  = self._extract_ee_positions(orig_pos_world[:, :agent_dim],  self.agent.end_effector)

        # ---------- Genesis visualization (driven by joint_cmd) ----------
        gs.init(theme="light", logging_level="warning")
        scene = gs.Scene(
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(0.0, 3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=30,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.Rasterizer(),
        )

        _ = scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.URDF(file=self.agent.urdf, collision=True, fixed=False))
        ball = scene.add_entity(
            gs.morphs.Sphere(
                radius=0.05,
            )
        )
        cam = scene.add_camera(
            res=(1280, 960),
            pos=(0.0, 3.5, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,
        )

        scene.build()
        robot.set_pos(np.array([0.0, 0.0, 0.42]))
        robot.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))  # identity quaternion

        # map joint names to local dof indices in Genesis
        dof_indices = self._extract_indices_for_robot(robot, self.agent.joint_name)

        seq_len = joint_cmd_np.shape[0]
        cam.start_recording()
        slow_factor = 1
        repeat_times = 3  # keep 1 by default; increase if you want to loop the clip

        for k in range(seq_len * slow_factor * repeat_times):
            t = (k // slow_factor) % seq_len
            robot.set_dofs_position(joint_cmd_np[t], dofs_idx_local=dof_indices)
            #ball.set_dofs_position(object_pos_world[t], dofs_idx_local=[0,1,2])
            scene.step()
            cam.render()

        cam.stop_recording(save_to_filename=save_path, fps=30)
        print(f"Saved sim video to: {save_path}")
