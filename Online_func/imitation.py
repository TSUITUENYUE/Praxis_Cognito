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
            processed_path=self.config.processed_path,
            source_path=demo,
            agent=self.agent
        )
        graph_x, orig_traj = dataset[index]
        graph_x = torch.tensor(graph_x, device=self.device)
        graph_x = graph_x.unsqueeze(0)  # Add batch dimension
        edge_index = build_edge_index(self.agent.fk_model, self.agent.end_effector, self.device)

        with torch.no_grad():
            recon_traj, joint_cmd, z, *_ = self.vae(graph_x, edge_index)


        recon_traj = recon_traj.squeeze(0).cpu().numpy()
        joint_cmd = joint_cmd.squeeze(0).cpu().numpy()
        orig_traj = orig_traj
        #print(orig_traj.shape)
        #print("loss:", abs(recon_traj - orig_traj).mean())
        plt.figure(figsize=(10, 6))
        plt.plot(abs(recon_traj - orig_traj).mean(axis=1))
        plt.title(f"loss")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        #plt.show()

        # Dimensions: object_dim=3, agent EE positions=12 (4 legs * 3D)
        agent_dim = 12
        orig_agent_pos = orig_traj[:, :agent_dim]
        orig_object_pos = orig_traj[:, agent_dim:]
        recon_agent_pos = recon_traj[:, :agent_dim]
        seq_len = orig_traj.shape[0]

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

        plane = scene.add_entity(gs.morphs.Plane())
        # Add imitation robot (blue material or default)
        imit_robot = scene.add_entity(gs.morphs.URDF(file=urdf_path,collision=True,))
        # Add demo robot (offset, red material for distinction if possible)
        demo_robot = scene.add_entity(gs.morphs.URDF(file=urdf_path,collision=True,))

        # Get end-effector links (calf links)
        ee_indices = self.agent.end_effector
        ee_links = [demo_robot.links[idx] for idx in ee_indices]  # List of RigidLink objects
        # Add objects as spheres
        imit_object = scene.add_entity(gs.morphs.Sphere(radius=0.05))
        demo_object = scene.add_entity(gs.morphs.Sphere(radius=0.05))


        scene.build()
        imit_robot.set_pos(np.array([0.0, 0.0, 0.42]))
        demo_robot.set_pos(np.array([0.0, 1.0, 0.42]))  # Offset to side
        demo_object.set_pos(np.array([1.0, 0.0, 0.0]))

        joint_names = self.agent.joint_name
        dof_indices = np.array([imit_robot.get_joint(name).dof_idx_local for name in joint_names])
        n_dofs = len(dof_indices)

        imit_robot.set_dofs_position(self.agent.init_angles, dofs_idx_local=dof_indices)
        demo_robot.set_dofs_position(self.agent.init_angles, dofs_idx_local=dof_indices)
        # Precompute demo joints using multi-link IK
        demo_joints = []
        for t in range(seq_len):
            # Split agent_pos into 4 leg positions (each 3D)
            agent_pos_t = orig_agent_pos[t]
            leg_poses = np.split(agent_pos_t, num_legs)  # List of [3] arrays

            # IK: poses as list of pos, quats repeated
            q = demo_robot.inverse_kinematics_multilink(
                links=ee_links,
                poss=leg_poses,
                quats=[fixed_quat] * num_legs
            )
            demo_joints.append(q[7:].cpu().numpy())

        demo_joints = np.array(demo_joints)  # [seq_len, 12]
        #print(demo_joints.shape)
        # Imitation joints from model (12 DOFs)

        '''
        imit_joints = []
        for t in range(seq_len):
            # Split agent_pos into 4 leg positions (each 3D)
            agent_pos_t = recon_agent_pos[t]
            leg_poses = np.split(agent_pos_t, num_legs)  # List of [3] arrays

            # IK: poses as list of pos, quats repeated
            p = imit_robot.inverse_kinematics_multilink(
                links=ee_links,
                poss=leg_poses,
                quats=[fixed_quat] * num_legs
            )
            imit_joints.append(p[7:].cpu().numpy())

        imit_joints = np.array(imit_joints)
        '''
        imit_joints = joint_cmd  # [seq_len, 12]
        #print(joint_cmd)
        # Simulation loop to animate both robots and objects
        i = 0
        while True:
            t = i // 5  # Slow down: 5 sim steps per traj frame
            t = min(t, seq_len - 1)

            # Control imitation robot with joint_cmd and move object
            imit_robot.control_dofs_position(imit_joints[t], dofs_idx)
            #imit_object.set_pos(recon_object_pos[t])

            # Control demo robot with IK joints and move object (offset)
            demo_robot.control_dofs_position(demo_joints[t], dofs_idx)
            #demo_object.set_pos(orig_object_pos[t] + np.array([1.0, 0.0, 0.0]))

            scene.step()
            i += 1
            if t == seq_len - 1 and i % 5 == 0:
                i = 0  # Loop animation