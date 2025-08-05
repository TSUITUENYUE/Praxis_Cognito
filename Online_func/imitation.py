import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from codebook import Codebook
from Pretrain.dataset import TrajectoryDataset
from Pretrain.utils import *
from torch.utils.data import DataLoader
import genesis as gs
import numpy as np
from matplotlib import pyplot as plt

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
        self.vae.eval()
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

    def visualize_in_sim(self, demo, index=0):
        # Load one sample from the dataset
        dataset = TrajectoryDataset(
            processed_path="./Pretrain/data/go2/25000 500000 5 30/preprocess.h5",
            agent=self.agent
        )

        dataset = TrajectoryDataset(
            processed_path=self.config.processed_path,
            source_path=demo,
            agent=self.agent
        )
        pos_mean = dataset.pos_mean
        pos_std = dataset.pos_std



        graph_x, orig_traj = dataset[index]
        graph_x = torch.tensor(graph_x, device=self.device)
        torch.set_printoptions(threshold=np.inf)
        #print(graph_x)
        graph_x = graph_x.unsqueeze(0)
        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.device)
        self.vae.eval()
        with torch.no_grad():
            recon_traj, joint_cmd, z, *_ = self.vae(graph_x, edge_index)

        recon_traj = recon_traj.squeeze(0).cpu().numpy()
        joint_cmd = joint_cmd.squeeze(0).cpu().numpy()
        orig_traj = orig_traj
        np.set_printoptions(threshold=np.inf)
        loss = ((recon_traj - orig_traj)**2).mean()
        #pos_std = torch.tensor([0.5, 0.6, 0.7])  # Example values

        print(loss)


        # plt.figure(figsize=(10, 6))
        # plt.plot(abs(recon_traj - orig_traj).mean(axis=1))
        # plt.title(f"loss")
        # plt.xlabel("Dimension 1")
        # plt.ylabel("Dimension 2")
        # plt.show()

        # Dimensions: object_dim=3, agent positions=126 (42 links * 3D)
        agent_dim = 126
        object_dim = 3
        orig_object_pos = orig_traj[:, agent_dim:]
        recon_object_pos = recon_traj[:, agent_dim:]
        seq_len = orig_traj.shape[0]

        def extract_ee_positions(traj, ee_indices):
            seq_len, _ = traj.shape
            ee_pos = np.zeros((seq_len, len(ee_indices) * 3))
            for j, idx in enumerate(ee_indices):
                start = idx * 3
                ee_pos[:, j * 3: (j * 3) + 3] = traj[:, start: start + 3]
            return ee_pos

        orig_agent_pos = extract_ee_positions(orig_traj[:, :agent_dim], self.agent.end_effector)
        recon_agent_pos = extract_ee_positions(recon_traj[:, :agent_dim], self.agent.end_effector)

        # URDF path from config
        urdf_path = self.agent.urdf
        num_legs = 4
        pos_per_leg = 3
        dofs_idx = np.arange(self.agent.n_dofs)  # 0 to 11 for 12 DOFs

        # Fixed quaternion for each foot (identity; adjust if needed for foot orientation)
        fixed_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z

        # Initialize scene
        gs.init(theme="light", logging_level='warning')

        NUM_ENVS = 1
        scene = gs.Scene(
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(0.0, 3.5, 2.5),
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

        plane = scene.add_entity(gs.morphs.Plane())
        # Add imitation robot (blue material or default)
        imit_robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, collision=True,fixed=True))
        # Add demo robot (offset, red material for distinction if possible)
        #demo_robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, collision=True))

        # Get end-effector links (calf links)
        #ee_indices = self.agent.end_effector
        #print(demo_robot.links)
        #ee_links = [demo_robot.links[idx] for idx in ee_indices]  # List of RigidLink objects
        # Add objects as spheres
        #imit_object = scene.add_entity(gs.morphs.Sphere(radius=0.05))
        #demo_object = scene.add_entity(gs.morphs.Sphere(radius=0.05))

        cam = scene.add_camera(
            res=(1280, 960),
            pos=(0.0, 3.5, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,
        )

        scene.build()
        imit_robot.set_pos(np.array([0.0, 0.0, 0.42]))
        #demo_robot.set_pos(np.array([0.0, 5.0, 0.42]))  # Offset to side

        joint_names = self.agent.joint_name
        dof_indices = np.array([imit_robot.get_joint(name).dof_idx_local for name in joint_names])
        print(dof_indices)
        n_dofs = len(dof_indices)
        # print(dof_indices)
        #imit_robot.set_dofs_position(self.agent.init_angles, dofs_idx_local=dof_indices)
        #demo_robot.set_dofs_position(self.agent.init_angles, dofs_idx_local=dof_indices)
        # Precompute demo joints using multi-link IK
        '''
        demo_joints = []
        for t in range(seq_len):
            agent_pos_t = orig_agent_pos[t]
            leg_poses = np.split(agent_pos_t, num_legs)
            p = demo_robot.inverse_kinematics_multilink(
                links=ee_links,
                poss=leg_poses,
                quats=[fixed_quat] * num_legs
            )
            demo_joints.append(p[7:].cpu().numpy())
        demo_joints = np.array(demo_joints)
        '''
        # Imitation joints from model (12 DOFs)
        imit_joints = joint_cmd  # [seq_len, 12]
        #print(joint_cmd)
        # Simulation loop to animate both robots and objects

        # print(recon_agent_pos)
        # print(orig_agent_pos)

        cam.start_recording()

        slow_factor = 1
        repeat_times = 3  # Repeat the sequence 3 times
        for ii in range(seq_len * slow_factor * repeat_times):
            t = (ii // slow_factor) % seq_len

            # Control imitation robot with joint_cmd and move object
            imit_robot.control_dofs_position(imit_joints[t], dofs_idx)
            # imit_object.set_pos(recon_object_pos[t])

            # Control demo robot with IK joints and move object (offset)
            # demo_robot.control_dofs_position(demo_joints[t], dofs_idx)
            # demo_object.set_pos(orig_object_pos[t] + np.array([0.0, 5.0, 0.0]))

            scene.step()
            cam.render()

        cam.stop_recording(save_to_filename='animation.mp4', fps=30)