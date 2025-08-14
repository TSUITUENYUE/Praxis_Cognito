import os, h5py, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Raw HDF5 must contain:
      - agent_trajectories: [N, T, n_dofs]
      - dof_vel:            [N, T, n_dofs]
      - obj_trajectories:   [N, T, 3]
      - obs:                [N, T, obs_dim]
      - actions:            [N, T, n_dofs]
      - (optional) valid_length: [N]   # first 'done' step per clip; if missing, assumes T

    We write processed:
      - graph_x:      z-scored positions of [links + object]   -> [N, T, num_nodes, 3]
      - joint_trajs:  raw joint angles                         -> [N, T, n_dofs]
      - joint_vels:   raw dq                                   -> [N, T, n_dofs]
      - obs:          raw                                      -> [N, T, obs_dim]
      - act:          raw post-tanh actions                    -> [N, T, n_dofs]
      - mask:         time mask (t < valid_length)             -> [N, T, 1] (bool)
      - valid_length: copied through for convenience           -> [N]
      - pos_mean/std: [3]
    __getitem__ returns (graph_x, obs, act, q, dq, mask)
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
        obs  = torch.from_numpy(self.h5_file['obs'][idx]).float()           # [T, obs_dim] (raw)
        act  = torch.from_numpy(self.h5_file['act'][idx]).float()           # [T, n_dofs]
        q    = torch.from_numpy(self.h5_file['joint_trajs'][idx]).float()   # [T, n_dofs]  (raw)
        dq   = torch.from_numpy(self.h5_file['joint_vels'][idx]).float()    # [T, n_dofs]  (raw rad/s)
        mask = torch.from_numpy(self.h5_file['mask'][idx]).squeeze(-1).float()  # [T] ★ NEW
        return gx, obs, act, q, dq, mask

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
            agent_trajs = f_in['agent_trajectories']  # [N,T,D]
            obj_trajs   = f_in['obj_trajectories']    # [N,T,3]
            N, T, _ = agent_trajs.shape
            num_links = len(fk.link_names)

            sum_pos = torch.zeros(3, device=device)
            sum_sq  = torch.zeros(3, device=device)
            count   = 0

            chunk = 1024
            for i in tqdm(range(0, N, chunk), desc="Stats pass"):
                j = min(i+chunk, N)
                q   = torch.from_numpy(agent_trajs[i:j]).to(device)     # [B,T,D]
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
            agent_trajs = f_in['agent_trajectories']
            obj_trajs   = f_in['obj_trajectories']
            obs_dset    = f_in['obs']
            act_dset    = f_in['actions']
            joint_vel   = f_in['dof_vel']
            # ★ NEW: valid_length may or may not be present
            has_vlen    = 'valid_length' in f_in
            valid_len_src = f_in['valid_length'][:] if has_vlen else None

            N, T, _ = agent_trajs.shape
            num_links = len(fk.link_names)
            num_nodes = num_links + 1
            obs_dim   = obs_dset.shape[-1]

            f_out.create_dataset('graph_x',      (N, T, num_nodes, 3), dtype='f4')
            f_out.create_dataset('joint_trajs',  (N, T, self.n_dofs),  dtype='f4')
            f_out.create_dataset('obs',          (N, T, obs_dim),      dtype='f4')
            f_out.create_dataset('joint_vels',   (N, T, self.n_dofs),  dtype='f4')
            f_out.create_dataset('act',          (N, T, self.n_dofs),  dtype='f4')
            f_out.create_dataset('mask',         (N, T, 1),            dtype='f4')
            f_out.create_dataset('pos_mean', data=pos_mean.cpu().numpy())
            f_out.create_dataset('pos_std',  data=pos_std.cpu().numpy())
            #f_out.create_dataset('valid_length', data=(valid_len_src if has_vlen else np.full((N,), T, dtype=np.int32)))  # ★ NEW

            # Precompute time index for mask building
            t_index = np.arange(T, dtype=np.int32)[None, :]  # [1, T] ★ NEW

            chunk = 512
            for i in tqdm(range(0, N, chunk), desc="Writing processed"):
                j = min(i+chunk, N)

                # FK -> positions
                q_np   = agent_trajs[i:j]                          # numpy view
                obj_np = obj_trajs[i:j]
                q      = torch.from_numpy(q_np).to(device)         # [B,T,D]
                obj    = torch.from_numpy(obj_np).to(device)       # [B,T,3]
                B = q.shape[0]
                qf = q.reshape(B*T, self.n_dofs)
                links_flat = fk(qf)                                # [B*T, links*3]
                links = links_flat.view(B, T, num_links, 3)
                obj = obj.unsqueeze(2)                             # [B,T,1,3]
                graph_x = torch.cat([links, obj], dim=2)           # [B,T,num_nodes,3]

                # normalize ONLY positions for loss
                graph_x = (graph_x - pos_mean) / pos_std

                # raw tensors (as float32)
                obs = torch.from_numpy(obs_dset[i:j]).float()
                q_raw  = torch.from_numpy(q_np).float()
                dq_raw = torch.from_numpy(joint_vel[i:j]).float()
                act    = torch.from_numpy(act_dset[i:j]).float()

                # ★ NEW: build mask from valid_length (vectorized for the block)
                if has_vlen:
                    vlen_block = valid_len_src[i:j].astype(np.int32)  # [B]
                else:
                    vlen_block = np.full((j - i,), T, dtype=np.int32)
                mask_block = float((t_index < vlen_block[:, None])[:, :, None])  # [B, T, 1] bool

                # write
                f_out['graph_x'][i:j]      = graph_x.cpu().numpy()
                f_out['joint_trajs'][i:j]  = q_raw.cpu().numpy()
                f_out['obs'][i:j]          = obs.cpu().numpy()
                f_out['joint_vels'][i:j]   = dq_raw.cpu().numpy()
                f_out['act'][i:j]          = act.cpu().numpy()
                f_out['mask'][i:j]         = mask_block