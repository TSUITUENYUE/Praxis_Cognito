import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
import genesis as gs


class TrajectoryDataset(Dataset):
    def __init__(self, processed_path, source_path=None, agent = None, force_reprocess=False,):
        super().__init__()
        self.mode = None
        self.processed_path = processed_path
        self.agent = agent
        self.n_dofs = self.agent.n_dofs
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

        with h5py.File(source_path, 'r') as f_in, h5py.File(self.processed_path, 'w') as f_out:
            agent_trajs = f_in['agent_trajectories']
            obj_trajs = f_in['obj_trajectories']
            num_samples, seq_len, _ = agent_trajs.shape

            f_out.create_dataset('graph_x', (num_samples, seq_len, num_nodes, 3), dtype='f4')
            f_out.create_dataset('orig_traj', (num_samples, seq_len, position_dim + 3), dtype='f4')

            chunk_size = 512
            for i in tqdm(range(0, num_samples, chunk_size), desc="Preprocessing"):
                end = min(i + chunk_size, num_samples)
                joint_angles_chunk = torch.from_numpy(agent_trajs[i:end]).to(device)
                obj_pos_chunk = torch.from_numpy(obj_trajs[i:end]).to(device)
                bs_chunk = joint_angles_chunk.shape[0]

                joint_angles_flat = joint_angles_chunk.reshape(bs_chunk * seq_len, n_dofs)
                link_pos_flat = fk_model(joint_angles_flat)
                link_pos = link_pos_flat.reshape(bs_chunk, seq_len, num_links, 3)
                obj_pos = obj_pos_chunk.unsqueeze(2)

                graph_x = torch.cat([link_pos, obj_pos], dim=2)
                orig_traj = torch.cat([link_pos.reshape(bs_chunk, seq_len, position_dim), obj_pos.squeeze(2)], dim=2)

                f_out['graph_x'][i:end] = graph_x.cpu().numpy()
                f_out['orig_traj'][i:end] = orig_traj.cpu().numpy()

        print("✅ Preprocessing complete.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'load':
            if self.h5_file is None:
                self.h5_file = h5py.File(self.processed_path, 'r')
            return self.h5_file['graph_x'][idx], self.h5_file['orig_traj'][idx]

