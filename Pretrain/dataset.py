import os, h5py, torch, numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Raw HDF5 must contain:
      - agent_trajectories: [N, T, n_dofs]
      - obj_trajectories:   [N, T, 3]
      - obs:                [N, T, obs_dim]
    We compute:
      - graph_x:  z-scored positions of [links + object]  -> [N, T, num_nodes, 3]
      - joint_trajs: raw joint angles                      -> [N, T, n_dofs]
      - joint_vels: raw dq (rad/s), dq[0]=0                -> [N, T, n_dofs]
      - obs:       raw                                     -> [N, T, obs_dim]
      - pos_mean/std: [3]
      - attrs: dt
    __getitem__ returns (graph_x, joint_trajs, obs, joint_vels)
    """
    def __init__(self, processed_path: str, source_path: str, agent, dt: float, force_reprocess: bool = False):
        super().__init__()
        assert agent is not None
        assert dt > 0
        self.processed_path = processed_path
        self.source_path = source_path
        self.agent = agent
        self.n_dofs = int(agent.n_dofs)
        self.dt = float(dt)
        self.h5_file = None

        if (not os.path.exists(processed_path)) or force_reprocess:
            self._preprocess()

        with h5py.File(self.processed_path, 'r') as f:
            gx = f['graph_x']
            self.num_samples, self.seq_len, self.num_nodes, _ = gx.shape
            self._pos_mean = torch.from_numpy(f['pos_mean'][:]).float()
            self._pos_std  = torch.from_numpy(f['pos_std'][:]).float()
            self.dt_attr   = float(f.attrs.get('dt', self.dt))

        print(f"✅ Dataset ready: N={self.num_samples}, T={self.seq_len}, nodes={self.num_nodes}, dt={self.dt_attr:.4f}s")

    def __len__(self): return self.num_samples

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.processed_path, 'r', libver='latest', swmr=True)
        gx  = torch.from_numpy(self.h5_file['graph_x'][idx]).float()      # [T, nodes, 3] (normalized)
        obs = torch.from_numpy(self.h5_file['obs'][idx]).float()          # [T, obs_dim] (raw)
        q   = torch.from_numpy(self.h5_file['joint_trajs'][idx]).float()  # [T, n_dofs]  (raw)
        dq  = torch.from_numpy(self.h5_file['joint_vels'][idx]).float()   # [T, n_dofs]  (raw rad/s)
        return gx, obs, q, dq

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
            assert all(k in f_in for k in ('agent_trajectories','obj_trajectories','obs'))
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
                q  = torch.from_numpy(agent_trajs[i:j]).to(device)      # [B,T,D]
                ob = torch.from_numpy(obj_trajs[i:j]).to(device)        # [B,T,3]
                B = q.shape[0]
                qf = q.reshape(B*T, self.n_dofs)
                links_flat = fk(qf)                                      # [B*T, links*3]
                links = links_flat.view(B, T, num_links, 3)
                all_pos = torch.cat([links.reshape(-1,3), ob.reshape(-1,3)], dim=0)
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

            N, T, _ = agent_trajs.shape
            num_links = len(fk.link_names)
            num_nodes = num_links + 1
            obs_dim   = obs_dset.shape[-1]

            f_out.create_dataset('graph_x',     (N, T, num_nodes, 3), dtype='f4')
            f_out.create_dataset('joint_trajs', (N, T, self.n_dofs),  dtype='f4')
            f_out.create_dataset('obs',         (N, T, obs_dim),      dtype='f4')
            f_out.create_dataset('joint_vels',  (N, T, self.n_dofs),  dtype='f4')
            f_out.create_dataset('pos_mean', data=pos_mean.cpu().numpy())
            f_out.create_dataset('pos_std',  data=pos_std.cpu().numpy())
            f_out.attrs['dt'] = self.dt

            chunk = 512
            for i in tqdm(range(0, N, chunk), desc="Writing processed"):
                j = min(i+chunk, N)

                # FK -> positions
                q  = torch.from_numpy(agent_trajs[i:j]).to(device)      # [B,T,D]
                ob = torch.from_numpy(obj_trajs[i:j]).to(device)        # [B,T,3]
                B = q.shape[0]
                qf = q.reshape(B*T, self.n_dofs)
                links_flat = fk(qf)                                      # [B*T, links*3]
                links = links_flat.view(B, T, num_links, 3)
                ob = ob.unsqueeze(2)                                     # [B,T,1,3]
                graph_x = torch.cat([links, ob], dim=2)                  # [B,T,num_nodes,3]

                # normalize ONLY positions for loss
                graph_x = (graph_x - pos_mean) / pos_std

                # raw obs
                obs = torch.from_numpy(obs_dset[i:j]).float()            # [B,T,O]

                # raw joint vels (dq[0]=0)
                q_cpu = torch.from_numpy(agent_trajs[i:j]).float()
                dq = torch.zeros_like(q_cpu)
                dq[:,1:] = (q_cpu[:,1:] - q_cpu[:, :-1]) / self.dt

                # write
                f_out['graph_x'][i:j]     = graph_x.cpu().numpy()
                f_out['joint_trajs'][i:j] = q_cpu.cpu().numpy()
                f_out['obs'][i:j]         = obs.cpu().numpy()
                f_out['joint_vels'][i:j]  = dq.cpu().numpy()

    # expose for your decoder init
    @property
    def pos_std(self):  return self._pos_std
    @property
    def pos_mean(self): return self._pos_mean
