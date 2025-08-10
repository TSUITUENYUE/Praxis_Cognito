import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
import genesis as gs


class TrajectoryDataset(Dataset):
    def __init__(self, processed_path, source_path=None, agent=None, force_reprocess=False):
        super().__init__()
        self.mode = None
        self.processed_path = processed_path
        self.agent = agent
        self.n_dofs = self.agent.n_dofs
        self.pos_mean = torch.zeros(3)
        self.pos_std = torch.ones(3)
        if os.path.exists(processed_path) and not force_reprocess:
            # --- MODE 1: LOAD EXISTING PREPROCESSED DATA ---
            self._init_load_mode()

        elif source_path:
            # --- MODE 2: PREPROCESS FROM SOURCE, THEN LOAD ---
            print(f"Preprocessing source file: {source_path}")
            self._run_preprocessing(source_path)
            self._init_load_mode()

    def _init_load_mode(self):
        """Initializes the dataset for loading from a preprocessed file."""
        self.mode = 'load'
        self.h5_file = None  # Will be opened by each worker in __getitem__
        with h5py.File(self.processed_path, 'r') as f:
            self.num_samples = f['graph_x'].shape[0]
            if 'pos_mean' in f and 'pos_std' in f:
                self.pos_mean = torch.from_numpy(f['pos_mean'][:])
                self.pos_std = torch.from_numpy(f['pos_std'][:])
            else:
                print("Warning: pos_mean and pos_std not found in the processed file. Using defaults (mean=0, std=1).")
        print(f"✅ Dataset in 'load' mode. Found {self.num_samples} samples.")

    def _run_preprocessing(self, source_path):
        """Performs the one-time preprocessing of the source dataset."""
        print("Starting data preprocessing...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fk_model = self.agent.fk_model.to(device)
        n_dofs = self.n_dofs
        num_links = len(fk_model.link_names)
        position_dim = num_links * 3
        num_nodes = num_links + 1

        # First pass: Compute mean and std (open f_in here)
        with h5py.File(source_path, 'r') as f_in:
            agent_trajs = f_in['agent_trajectories']
            obj_trajs = f_in['obj_trajectories']
            num_samples, seq_len, joint_dim = agent_trajs.shape

            sum_pos = torch.zeros(3, device=device)
            sum_sq = torch.zeros(3, device=device)
            count = 0
            chunk_size = 8192
            for i in tqdm(range(0, num_samples, chunk_size), desc="Computing stats"):
                end = min(i + chunk_size, num_samples)
                joint_angles_chunk = torch.from_numpy(agent_trajs[i:end]).to(device)
                obj_pos_chunk = torch.from_numpy(obj_trajs[i:end]).to(device)
                bs_chunk = joint_angles_chunk.shape[0]

                joint_angles_flat = joint_angles_chunk.reshape(bs_chunk * seq_len, n_dofs)
                link_pos_flat = fk_model(joint_angles_flat)
                link_pos = link_pos_flat.reshape(bs_chunk, seq_len, num_links, 3)

                # All positions: links + objs
                all_pos = torch.cat([link_pos.reshape(-1, 3), obj_pos_chunk.reshape(-1, 3)], dim=0)

                sum_pos += all_pos.sum(dim=0)
                sum_sq += (all_pos ** 2).sum(dim=0)
                count += all_pos.size(0)

        pos_mean = sum_pos / count
        pos_var = (sum_sq / count) - (pos_mean ** 2)
        pos_std = torch.sqrt(pos_var) + 1e-6  # Avoid division by zero

        # Second pass: Normalize and save (reopen f_in here)
        with h5py.File(source_path, 'r') as f_in, h5py.File(self.processed_path, 'w') as f_out:
            agent_trajs = f_in['agent_trajectories']
            obj_trajs = f_in['obj_trajectories']

            f_out.create_dataset('graph_x', (num_samples, seq_len, num_nodes, 3), dtype='f4')
            f_out.create_dataset('orig_traj', (num_samples, seq_len, position_dim + 3), dtype='f4')
            f_out.create_dataset('joint_trajs', data=agent_trajs)
            f_out.create_dataset('pos_mean', data=pos_mean.cpu().numpy())
            f_out.create_dataset('pos_std', data=pos_std.cpu().numpy())

            for i in tqdm(range(0, num_samples, chunk_size), desc="Normalizing and saving"):
                end = min(i + chunk_size, num_samples)
                joint_angles_chunk = torch.from_numpy(agent_trajs[i:end]).to(device)
                torch.set_printoptions(threshold=np.inf)

                obj_pos_chunk = torch.from_numpy(obj_trajs[i:end]).to(device)
                bs_chunk = joint_angles_chunk.shape[0]

                joint_angles_flat = joint_angles_chunk.reshape(bs_chunk * seq_len, n_dofs)
                link_pos_flat = fk_model(joint_angles_flat)

                link_pos = link_pos_flat.reshape(bs_chunk, seq_len, num_links, 3)
                obj_pos = obj_pos_chunk.unsqueeze(2)

                graph_x = torch.cat([link_pos, obj_pos], dim=2)
                orig_traj = torch.cat([link_pos.reshape(bs_chunk, seq_len, position_dim), obj_pos.squeeze(2)], dim=2)

                # Z-score Normalize

                graph_x_norm = (graph_x - pos_mean) / pos_std

                orig_traj_reshaped = orig_traj.reshape(bs_chunk, seq_len, num_links + 1, 3)
                orig_traj_norm_reshaped = (orig_traj_reshaped - pos_mean) / pos_std
                orig_traj_norm = orig_traj_norm_reshaped.reshape(bs_chunk, seq_len, position_dim + 3)

                f_out['graph_x'][i:end] = graph_x_norm.cpu().numpy()
                f_out['orig_traj'][i:end] = orig_traj_norm.cpu().numpy()
                '''
                f_out['graph_x'][i:end] = graph_x.cpu().numpy()
                f_out['orig_traj'][i:end] = orig_traj.cpu().numpy()
                '''
            #f_out['agent_trajs'] = agent_trajs
        print("✅ Preprocessing complete.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'load':
            if self.h5_file is None:
                self.h5_file = h5py.File(self.processed_path, 'r')
            return self.h5_file['graph_x'][idx], self.h5_file['orig_traj'][idx], self.h5_file['joint_trajs'][idx]