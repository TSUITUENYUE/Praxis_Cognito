import os, h5py, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Raw HDF5 must contain:
      - dof_pos:            [N, T, n_dofs]
      - dof_vel:            [N, T, n_dofs]
      - base_pos:           [N, T, 3]
      - base_vel:           [N, T, 3]
      - base_ang:           [N, T, 3]
      - obs:                [N, T, obs_dim]
      - actions:            [N, T, n_dofs]
      - valid_length: [N]   # first 'done' step per clip; if missing, assumes T

    We write processed:
      - graph_x:      z-scored positions of [links + object]   -> [N, T, num_nodes, 3]
      - joint_trajs:  raw joint angles                         -> [N, T, n_dofs]
      - joint_vels:   raw dq                                   -> [N, T, n_dofs]
      - base_pos:     raw                                      -> [N, T, 3]
      - base_vel:     raw                                      -> [N, T, 3]
      - base_ang:     raw                                      -> [N, T, 3]
      - obs:          raw                                      -> [N, T, obs_dim]
      - act:          raw post-tanh actions                    -> [N, T, n_dofs]
      - mask:         time mask (t < valid_length)             -> [N, T, 1] (bool)
      - pos_mean/std: [3]
    __getitem__ returns (graph_x, q, dq, p, dp, w, obs, act, mask)
    """
    def __init__(self, processed_path: str, source_path: str, agent, force_reprocess: bool = False):
        super().__init__()
        assert agent is not None
        self.processed_path = processed_path
        self.source_path = source_path
        self.agent = agent
        self.n_dofs = int(agent.n_dofs)
        self.h5_file = None

        if (not os.path.exists(processed_path)) or force_reprocess:
            self._preprocess()

        with h5py.File(self.processed_path, 'r') as f:
            gx = f['graph_x']
            self.num_samples, self.seq_len, self.num_nodes, _ = gx.shape
            self._pos_mean = torch.from_numpy(f['pos_mean'][:]).float()
            self._pos_std  = torch.from_numpy(f['pos_std'][:]).float()

        print(f"✅ Dataset ready: N={self.num_samples}, T={self.seq_len}, nodes={self.num_nodes}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.processed_path, 'r', libver='latest', swmr=True)
        gx   = torch.from_numpy(self.h5_file['graph_x'][idx]).float()       # [T, nodes, 3] (normalized)

        q = torch.from_numpy(self.h5_file['joint_trajs'][idx]).float()  # [T, n_dofs]  (raw)
        dq = torch.from_numpy(self.h5_file['joint_vels'][idx]).float()  # [T, n_dofs]  (raw rad/s)

        p = torch.from_numpy(self.h5_file['base_pos'][idx]).float()     # [T, 3]  (raw)
        dp = torch.from_numpy(self.h5_file['base_vel'][idx]).float()    # [T, 3]  (raw)
        w = torch.from_numpy(self.h5_file['base_ang'][idx]).float()     # [T, 3]  (raw)

        u = torch.from_numpy(self.h5_file['ball_pos'][idx]).float()     # [T, 3]  (raw)
        du = torch.from_numpy(self.h5_file['ball_vel'][idx]).float()    # [T, 3]  (raw)
        v = torch.from_numpy(self.h5_file['ball_ang'][idx]).float()     # [T, 3]  (raw)

        obs  = torch.from_numpy(self.h5_file['obs'][idx]).float()           # [T, obs_dim] (raw)
        act  = torch.from_numpy(self.h5_file['act'][idx]).float()           # [T, n_dofs]

        mask = torch.from_numpy(self.h5_file['mask'][idx]).squeeze(-1).float()  # [T] ★ NEW
        return gx, q, dq, p, dp, w, u, du, v, obs, act, mask

    @property
    def pos_mean(self): return self._pos_mean
    @property
    def pos_std(self):  return self._pos_std

    # ---------------- internals ----------------
    def _preprocess(self):
        assert os.path.exists(self.source_path), f"Missing raw file: {self.source_path}"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fk = self.agent.fk_model.to(device)

        # ---- Pass 1: position stats (links + object) ----
        with h5py.File(self.source_path, 'r') as f_in:
            dof_pos = f_in['dof_pos']  # [N,T,D]
            obj_trajs = f_in['obs'][:,:,-6:-3]    # [N,T,3]
            N, T, _ = dof_pos.shape
            num_links = len(fk.link_names)

            sum_pos = torch.zeros(3, device=device)
            sum_sq  = torch.zeros(3, device=device)
            count   = 0

            chunk = 1024
            for i in tqdm(range(0, N, chunk), desc="Stats pass"):
                j = min(i+chunk, N)
                q   = torch.from_numpy(dof_pos[i:j]).to(device)     # [B,T,D]
                obj = torch.from_numpy(obj_trajs[i:j]).to(device)       # [B,T,3]
                B = q.shape[0]
                qf = q.reshape(B*T, self.n_dofs)
                links_flat = fk(qf)                                     # [B*T, links*3]
                links = links_flat.view(B, T, num_links, 3)
                all_pos = torch.cat([links.reshape(-1,3), obj.reshape(-1,3)], dim=0)
                sum_pos += all_pos.sum(dim=0)
                sum_sq  += (all_pos**2).sum(dim=0)
                count   += all_pos.size(0)

            pos_mean = sum_pos / count
            pos_var  = (sum_sq / count) - (pos_mean**2)
            pos_std  = torch.sqrt(pos_var.clamp_min(1e-12)) + 1e-6

        # ---- Pass 2: write processed ----
        with h5py.File(self.source_path, 'r') as f_in, h5py.File(self.processed_path, 'w') as f_out:
            dof_pos = f_in['dof_pos']  # [N,T,D]
            dof_vel = f_in['dof_vel']  # [N,T,D]
            base_pos_ds = f_in['base_pos']  # [N,T,3]
            base_vel_ds = f_in['base_vel']  # [N,T,3]
            base_ang_ds = f_in['base_ang']  # [N,T,3]
            ball_pos_ds = f_in['ball_pos']  # [N,T,3]
            ball_vel_ds = f_in['ball_vel']  # [N,T,3]
            ball_ang_ds = f_in['ball_ang']  # [N,T,3]

            obs_ds = f_in['obs']  # [N,T,obs_dim]
            act_key = 'act' if 'act' in f_in else 'actions'
            act_ds = f_in[act_key]  # [N,T,n_dofs]

            N, T, _ = dof_pos.shape
            num_links = len(fk.link_names)
            num_nodes = num_links + 1
            obs_dim = obs_ds.shape[-1]

            # valid_length is optional
            if 'valid_length' in f_in:
                valid_len = f_in['valid_length'][:].astype(np.int32)
            else:
                valid_len = np.full((N,), T, dtype=np.int32)

            # Create empty datasets; we'll fill them in chunks
            f_out.create_dataset('graph_x', (N, T, num_nodes, 3), dtype='f4')
            f_out.create_dataset('joint_trajs', (N, T, self.n_dofs), dtype='f4')
            f_out.create_dataset('joint_vels', (N, T, self.n_dofs), dtype='f4')

            f_out.create_dataset('base_pos', (N, T, 3), dtype='f4')
            f_out.create_dataset('base_vel', (N, T, 3), dtype='f4')
            f_out.create_dataset('base_ang', (N, T, 3), dtype='f4')

            f_out.create_dataset('ball_pos', (N, T, 3), dtype='f4')
            f_out.create_dataset('ball_vel', (N, T, 3), dtype='f4')
            f_out.create_dataset('ball_ang', (N, T, 3), dtype='f4')

            f_out.create_dataset('obs', (N, T, obs_dim), dtype='f4')
            f_out.create_dataset('act', (N, T, self.n_dofs), dtype='f4')
            f_out.create_dataset('mask', (N, T, 1), dtype='f4')

            f_out.create_dataset('pos_mean', data=pos_mean.cpu().numpy())
            f_out.create_dataset('pos_std', data=pos_std.cpu().numpy())

            t_index = np.arange(T, dtype=np.int32)[None, :]  # [1,T]

            chunk = 512
            for i in tqdm(range(0, N, chunk), desc="Writing processed"):
                j = min(i + chunk, N)

                # --- FK -> link positions (torch) ---
                q_np = dof_pos[i:j]  # (B,T,D) numpy
                q = torch.from_numpy(q_np).to(device)  # torch
                B = q.shape[0]
                qf = q.reshape(B * T, self.n_dofs)
                links_flat = fk(qf)  # (B*T, links*3)
                links = links_flat.view(B, T, num_links, 3)

                # Object positions from obs, streamed per chunk
                obj_np = obs_ds[i:j, :, -6:-3]  # (B,T,3) numpy
                obj = torch.from_numpy(obj_np).to(device).unsqueeze(2)  # (B,T,1,3)

                # Concatenate and normalize
                graph_x = torch.cat([links, obj], dim=2)  # (B,T,num_nodes,3)
                graph_x = ((graph_x - pos_mean) / pos_std).cpu().numpy().astype('f4')

                # --- Write outputs (numpy needed; h5py will cast) ---
                f_out['graph_x'][i:j] = graph_x
                f_out['joint_trajs'][i:j] = q_np.astype('f4')
                f_out['joint_vels'][i:j] = dof_vel[i:j].astype('f4')

                f_out['base_pos'][i:j] = base_pos_ds[i:j].astype('f4')
                f_out['base_vel'][i:j] = base_vel_ds[i:j].astype('f4')
                f_out['base_ang'][i:j] = base_ang_ds[i:j].astype('f4')
                f_out['ball_pos'][i:j] = ball_pos_ds[i:j].astype('f4')
                f_out['ball_vel'][i:j] = ball_vel_ds[i:j].astype('f4')
                f_out['ball_ang'][i:j] = ball_ang_ds[i:j].astype('f4')

                f_out['obs'][i:j] = obs_ds[i:j].astype('f4')
                f_out['act'][i:j] = act_ds[i:j].astype('f4')

                vlen_block = valid_len[i:j]  # (B,)
                mask_block = (t_index < vlen_block[:, None])[:, :, None].astype('f4')  # (B,T,1)
                f_out['mask'][i:j] = mask_block
