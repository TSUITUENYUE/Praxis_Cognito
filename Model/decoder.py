import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, object_dim, joint_dim, agent, hidden_dim, pos_mean, pos_std,
                 time_bands: int = 6, include_linear_time: bool = True):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.joint_dim = joint_dim
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.agent = agent

        self.fk_model = self.agent.fk_model

        # === Normalization buffers (stay on device with the module) ===
        self.register_buffer("pos_mean", pos_mean)   # [3] (or [num_nodes,3] if you switch later)
        self.register_buffer("pos_std",  pos_std)    # [3]

        # === Kinematics / limits ===
        device = pos_mean.device
        jl = torch.as_tensor(self.agent.joint_limits_lower, device=device, dtype=torch.float32)
        ju = torch.as_tensor(self.agent.joint_limits_upper, device=device, dtype=torch.float32)
        self.register_buffer("joint_lower", jl)
        self.register_buffer("joint_upper", ju)
        self.register_buffer("joint_range", (ju - jl) / 2.0)
        self.register_buffer("joint_mean",  (ju + jl) / 2.0)

        # Start pose from robot init angles
        init_angles = torch.as_tensor(self.agent.init_angles, device=device, dtype=torch.float32)  # [DoF]
        self.register_buffer("init_angles", init_angles)

        # === Time embedding (NeRF-style Fourier features) ===
        self.time_bands = time_bands
        self.include_linear_time = include_linear_time
        self.time_dim = (2 * time_bands) + (1 if include_linear_time else 0)

        # === Sizes ===
        self.num_links = len(self.fk_model.link_names)
        self.pos_dim = (self.num_links + 1) * 3  # links + object

        # === Recurrent core ===
        lstm_input_dim = self.joint_dim + self.latent_dim + self.time_dim
        self.rnn = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)

        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.z_to_cell   = nn.Linear(latent_dim, hidden_dim)

        # === Heads ===
        # Object position head (unchanged)
        self.mlp_object = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.object_dim)
        )

        # Δq head: predict per-step joint delta in [-1,1], then scale (see self.delta_scale)
        self.mlp_delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),
        )

        # Per-joint step scale (learned), initialized small so motion starts safe
        self.delta_scale = nn.Parameter(torch.ones(self.joint_dim, device=device) * 0.05)  # ~0.05 rad

        # Heteroscedastic variance head (for NLL on positions)
        self.var_head = nn.Linear(hidden_dim, self.pos_dim)
        nn.init.zeros_(self.var_head.weight)
        # Bias set so initial sigma ~ 0.1 after sigmoid mapping
        nn.init.constant_(self.var_head.bias, self._logit_for_sigma(0.10, smin=0.05, smax=0.5))

        # Sigma bounds for mapping with sigmoid
        self.sigma_min = 5e-2
        self.sigma_max = 5e-1

    @staticmethod
    def _logit_for_sigma(target_sigma, smin, smax):
        t = (target_sigma - smin) / (smax - smin)
        t = min(max(float(t), 1e-6), 1.0 - 1e-6)
        return math.log(t / (1.0 - t))

    def _time_features(self, T, device):
        # t in [0,1]
        t = torch.linspace(0.0, 1.0, T, device=device)  # [T]
        feats = []
        if self.include_linear_time:
            feats.append(t)
        for k in range(self.time_bands):
            f = 2.0 ** k
            feats.append(torch.sin(2 * math.pi * f * t))
            feats.append(torch.cos(2 * math.pi * f * t))
        return torch.stack(feats, dim=-1)  # [T, time_dim]

    def forward(self, z, teacher_joints=None):
        """
        Inputs:
          z: [B, Z]
          teacher_joints: [B, T, DoF] or None
        Returns:
          graph_x_mu: [B, T, pos_dim]  (normalized positions for AE loss)
          joint_traj: [B, T, DoF]      (unnormalized joint angles)
          log_sigma_pos: [B, T, pos_dim]
        """
        B = z.size(0)
        dev = z.device

        # RNN initial state from z
        h = self.z_to_hidden(z).unsqueeze(0)  # [1,B,H]
        c = self.z_to_cell(z).unsqueeze(0)    # [1,B,H]

        # Time features
        time_feats = self._time_features(self.seq_len, dev)            # [T, time_dim]
        time_feats = time_feats.unsqueeze(0).expand(B, -1, -1)         # [B, T, time_dim]

        # Start from teacher's first pose (if provided) else robot init pose
        if self.training and (teacher_joints is not None):
            prev_joints = teacher_joints[:, 0, :]                      # [B, DoF]
        else:
            prev_joints = self.init_angles.unsqueeze(0).expand(B, -1)  # [B, DoF]

        all_joint_outputs = []
        all_object_outputs = []
        all_hidden = []

        for t in range(self.seq_len):
            # Normalize previous joints for input (keep integration in joint space)
            prev_joints_norm = (prev_joints - self.joint_mean) / self.joint_range  # [B, DoF]

            lstm_input = torch.cat([prev_joints_norm, z, time_feats[:, t, :]], dim=-1).unsqueeze(1)  # [B,1,inp]
            output_h, (h, c) = self.rnn(lstm_input, (h, c))      # output_h: [B,1,H]
            h_t = output_h.squeeze(1)                            # [B,H]
            all_hidden.append(h_t)

            # Predict object position features (unnormalized world space via FK later)
            current_object = self.mlp_object(h_t)                # [B,3]

            # Predict Δq and integrate
            raw_delta = self.mlp_delta(h_t)  # [B, DoF], unconstrained
            delta_q = raw_delta * self.delta_scale  # soft step sizing
            current_joints = prev_joints + delta_q  # integrate
            # Clamp to joint limits
            clamped = torch.max(torch.min(current_joints, self.joint_upper), self.joint_lower)
            current_for_input = current_joints + (clamped - current_joints).detach()

            all_joint_outputs.append(current_joints)
            all_object_outputs.append(current_object)

            # Next-step input
            if self.training and (teacher_joints is not None):
                if t + 1 < self.seq_len:
                    prev_joints = teacher_joints[:, t + 1, :]
                else:
                    prev_joints = current_for_input.detach()
            else:
                prev_joints = current_for_input.detach()
        # Stack sequences
        joint_traj  = torch.stack(all_joint_outputs, dim=1)   # [B,T,DoF]
        object_traj = torch.stack(all_object_outputs, dim=1)  # [B,T,3]
        H_seq       = torch.stack(all_hidden, dim=1)          # [B,T,H]

        # FK to link positions
        joint_traj_flat   = joint_traj.reshape(B * self.seq_len, self.joint_dim)
        position_out_flat = self.fk_model(joint_traj_flat)            # [B*T, num_links*3]
        agent_traj        = position_out_flat.view(B, self.seq_len, -1)  # [B,T,num_links*3]

        # Combine agent (FK) + object
        combined_traj = torch.cat([agent_traj, object_traj], dim=-1)     # [B,T,pos_dim]
        combined_reshaped = combined_traj.view(B, self.seq_len, self.num_links + 1, 3)
        combined_norm = (combined_reshaped - self.pos_mean) / self.pos_std
        graph_x_mu = combined_norm.view(B, self.seq_len, -1)              # [B,T,pos_dim]

        # Heteroscedastic σ prediction (same shape as graph_x_mu)
        sigma_raw = self.var_head(H_seq)                                  # [B,T,pos_dim]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        log_sigma_pos = torch.log(sigma)

        return graph_x_mu, joint_traj, log_sigma_pos
