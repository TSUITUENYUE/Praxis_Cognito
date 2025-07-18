import genesis as gs
import taichi as ti
import numpy as np
import time
import torch
import h5py
import os
import hydra
from omegaconf import DictConfig


def generate(cfg: DictConfig):
    # --- Configuration for Training ---
    NUM_ENVS = cfg.dataset.num_envs  # Increase for more parallel data generation
    EPISODES_TO_COLLECT = cfg.dataset.episodes  # The number of episodes to generate
    MAX_EPISODE_SECONDS = cfg.dataset.max_episode_seconds
    FRAME_RATE = cfg.dataset.frame_rate
    AGENT = cfg.agent.name
    path = f"./Pretrain/data/{AGENT}/{NUM_ENVS} {EPISODES_TO_COLLECT} {MAX_EPISODE_SECONDS} {FRAME_RATE}"
    if not os.path.exists(path): os.makedirs(path)
    SAVE_FILENAME = f"{path}/{NUM_ENVS} {EPISODES_TO_COLLECT} {MAX_EPISODE_SECONDS} {FRAME_RATE}.h5"

    # --- Initialize Genesis Simulator ---
    gs.init(theme="light", logging_level='warning')

    scene = gs.Scene(
        show_viewer=NUM_ENVS < 128,
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

    # --- Add Entities ---
    plane = scene.add_entity(gs.morphs.Plane())
    agent = scene.add_entity(
        gs.morphs.URDF(file=cfg.agent.urdf, collision=True),
    )
    obj = scene.add_entity(
        gs.morphs.Sphere(radius=0.05, collision=True),
    )
    obj.fixed = False
    # --- Build the Scene ---
    scene.build(n_envs=NUM_ENVS, env_spacing=(5.0, 5.0))

    # --- Joint and DOF Setup ---
    joint_names = cfg.agent.joint_name
    dof_indices = np.array([agent.get_joint(name).dof_idx_local for name in joint_names])
    n_dofs = len(dof_indices)

    # --- Stratified Sampling Logic ---
    # This part of the logic remains the same
    joint_limits = np.array([agent.get_joint(name).dofs_limit[0] for name in joint_names])
    STRATIFICATION_BINS = cfg.dataset.stratification_bins
    stratified_grid = np.array([
        np.linspace(limit[0], limit[1], STRATIFICATION_BINS) for limit in joint_limits
    ])


    def sample_random_poses():
        bin_indices = np.random.randint(0, STRATIFICATION_BINS, size=(NUM_ENVS, n_dofs))
        target_poses = stratified_grid[np.arange(n_dofs)[:, np.newaxis], bin_indices.T].T
        return target_poses

    # --- Reset Functions ---
    def reset_agent_and_object():
        """Resets the agent and object state for all environments."""
        # Reset agent
        initial_pos_agent = np.tile(np.array(cfg.dataset.agent_pos), (NUM_ENVS, 1))
        agent.set_pos(initial_pos_agent)

        initial_joint_angles = np.tile(np.array(cfg.agent.init_angles), (NUM_ENVS, 1))
        agent.set_dofs_position(initial_joint_angles, dofs_idx_local=dof_indices)

        initial_vel_agent = np.zeros((NUM_ENVS, n_dofs))
        agent.set_dofs_velocity(initial_vel_agent, dofs_idx_local=dof_indices)


        # Reset object randomly
        pos_low = np.array(cfg.dataset.obj_pos_low)
        pos_high = np.array(cfg.dataset.obj_pos_high)
        random_pos = np.random.uniform(low=pos_low, high=pos_high, size=(NUM_ENVS, 3))
        obj.set_pos(random_pos)

        vel_low = np.array(cfg.dataset.obj_vel_low)
        vel_high = np.array(cfg.dataset.obj_vel_high)
        random_vel = np.random.uniform(low=vel_low, high=vel_high, size=(NUM_ENVS, 3))
        obj.set_dofs_velocity(random_vel, [0, 1, 2])


    # --- Main Continuous Simulation Loop ---
    max_episode_len = int(MAX_EPISODE_SECONDS * FRAME_RATE)
    total_episodes_saved = 0

    # Pre-allocate trajectory buffers on the GPU
    agent_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, n_dofs), device='cuda')
    obj_traj_buffer = torch.zeros((max_episode_len, NUM_ENVS, 3), device='cuda')  # x,y,z pos

    print(f"Starting data generation. Target: {EPISODES_TO_COLLECT} episodes.")
    if os.path.exists(SAVE_FILENAME):
        os.remove(SAVE_FILENAME)
        print(f"Removed existing file: '{SAVE_FILENAME}'")

    start_time = time.time()

    # Open HDF5 file once with a 'with' statement to ensure it's properly closed
    with h5py.File(SAVE_FILENAME, 'w') as f:
        print(f"Opened HDF5 file '{SAVE_FILENAME}' for writing.")

        # Create large datasets for all episodes (improvement for I/O speed)
        agent_ds = f.create_dataset('agent_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, n_dofs), dtype='float32', compression="gzip")
        obj_ds = f.create_dataset('obj_trajectories', shape=(EPISODES_TO_COLLECT, max_episode_len, 3), dtype='float32', compression="gzip")

        # Initial full reset for all environments
        scene.reset()
        reset_agent_and_object()

        while total_episodes_saved < EPISODES_TO_COLLECT:
            # --- BATCH COLLECTION LOOP ---
            # Precompute all bin indices and poses for the entire batch (improvement for loop speed)
            all_bin_indices = np.random.randint(0, STRATIFICATION_BINS, size=(max_episode_len, NUM_ENVS, n_dofs))
            all_target_poses = stratified_grid[np.arange(n_dofs)[None, None, :], all_bin_indices]

            all_targets = torch.from_numpy(all_target_poses).to('cuda')

            # Run one full batch of episodes for max_episode_len steps
            for t in range(max_episode_len):
                # Store data directly into the GPU buffer at the current timestep
                agent_traj_buffer[t] = agent.get_dofs_position(dof_indices)
                obj_traj_buffer[t] = obj.get_pos()

                # Control and Simulation Step
                agent.control_dofs_position(all_targets[t], dof_indices)
                scene.step()

            # --- BATCH PROCESSING AND SAVING ---
            # Transfer the entire batch of trajectories from GPU to CPU
            agent_data_batch_np = agent_traj_buffer.cpu().numpy()
            obj_data_batch_np = obj_traj_buffer.cpu().numpy()

            # Write batch as slices to the large datasets
            start_idx = total_episodes_saved
            end_idx = min(start_idx + NUM_ENVS, EPISODES_TO_COLLECT)
            num_this_batch = end_idx - start_idx

            agent_ds[start_idx:end_idx] = np.transpose(agent_data_batch_np[:, :num_this_batch, :], (1, 0, 2))
            obj_ds[start_idx:end_idx] = np.transpose(obj_data_batch_np[:, :num_this_batch, :], (1, 0, 2))

            total_episodes_saved += num_this_batch

            print(f"  ...Collected and saved episodes up to {total_episodes_saved}/{EPISODES_TO_COLLECT}")

            # --- RESET FOR NEXT BATCH ---
            if total_episodes_saved < EPISODES_TO_COLLECT:
                scene.reset()
                reset_agent_and_object()

    # --- Finalize ---
    end_time = time.time()
    duration = end_time - start_time
    print("\nData generation complete.")
    print(f"Collected {total_episodes_saved} episodes across {NUM_ENVS} parallel environments.")
    print(f"Dataset saved to '{SAVE_FILENAME}'")
    print(f"Total WALL-CLOCK time taken: {duration:.2f} seconds.")

