import torch
import torch.nn as nn
import math
from typing import Optional

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        object_dim: int,
        joint_dim: int,
        agent,
        hidden_dim: int,
        pos_mean: torch.Tensor,
        pos_std: torch.Tensor,
        time_bands: int = 6,
        include_linear_time: bool = True,
        obs_dim: int = 0,            # optional external observation dim (e.g., RL obs)
        obs_embed: int = 128,        # embed obs -> obs_embed
        action_scale: float = 0.25,  # match Go2Env: target = a * action_scale + default
        track_alpha: float = 1.0,    # 1.0 == instant track; <1.0 smooths toward target
        predict_object: bool = True  # keep object head unless you pass object_override
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.joint_dim = joint_dim
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.agent = agent
        self.predict_object = predict_object

        self.fk_model = self.agent.fk_model

        # === Normalization buffers ===
        self.register_buffer("pos_mean", pos_mean)  # [3] or broadcastable
        self.register_buffer("pos_std",  pos_std)   # [3]

        # === Joint limits / defaults ===
        device = pos_mean.device
        jl = torch.as_tensor(self.agent.joint_limits_lower, device=device, dtype=torch.float32)
        ju = torch.as_tensor(self.agent.joint_limits_upper, device=device, dtype=torch.float32)
        self.register_buffer("joint_lower", jl)
        self.register_buffer("joint_upper", ju)
        self.register_buffer("joint_range", (ju - jl) / 2.0)
        self.register_buffer("joint_mean",  (ju + jl) / 2.0)

        init_angles = torch.as_tensor(self.agent.init_angles, device=device, dtype=torch.float32)
        self.register_buffer("default_dof_pos", init_angles)  # used like Go2Env default
        self.action_scale = action_scale
        self.track_alpha = track_alpha

        # === Time embedding (Fourier) ===
        self.time_bands = time_bands
        self.include_linear_time = include_linear_time
        self.time_dim = (2 * time_bands) + (1 if include_linear_time else 0)

        # === Sizes ===
        self.num_links = len(self.fk_model.link_names)
        self.pos_dim = (self.num_links + 1) * 3  # links + object

        # === Obs projection (optional) ===
        self.obs_dim = obs_dim
        self.obs_embed = obs_embed
        self.obs_proj = None
        if obs_dim > 0:
            self.obs_proj = nn.Sequential(
                nn.Linear(obs_dim, obs_embed),
                nn.LayerNorm(obs_embed),
                nn.ReLU(),
            )

        # === Recurrent core ===
        lstm_input_dim = self.joint_dim + self.latent_dim + self.time_dim + (obs_embed if obs_dim > 0 else 0)
        self.rnn = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.z_to_cell   = nn.Linear(latent_dim, hidden_dim)

        # === Heads ===
        # Policy head -> actions in [-1, 1]
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),
            nn.Tanh()
        )

        # Optional object head (kept for AE compatibility)
        if self.predict_object:
            self.mlp_object = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.object_dim)
            )

        # Heteroscedastic variance head (for NLL on positions)
        self.var_head = nn.Linear(hidden_dim, self.pos_dim)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, self._logit_for_sigma(0.10, smin=0.05, smax=0.5))

        self.sigma_min = 5e-2
        self.sigma_max = 5e-1

    @staticmethod
    def _logit_for_sigma(target_sigma, smin, smax):
        t = (target_sigma - smin) / (smax - smin)
        t = min(max(float(t), 1e-6), 1.0 - 1e-6)
        return math.log(t / (1.0 - t))

    def _time_features(self, T, device):
        t = torch.linspace(0.0, 1.0, T, device=device)
        feats = []
        if self.include_linear_time:
            feats.append(t)
        for k in range(self.time_bands):
            f = 2.0 ** k
            feats.append(torch.sin(2 * math.pi * f * t))
            feats.append(torch.cos(2 * math.pi * f * t))
        return torch.stack(feats, dim=-1)  # [T, time_dim]

    @torch.no_grad()
    def act(self, z: torch.Tensor, obs_t: Optional[torch.Tensor], prev_joints: torch.Tensor):
        """
        One step policy inference (no gradients). Matches Go2Env scaling.
        z: [B, Z]; obs_t: [B, obs_dim] or None; prev_joints: [B, DoF]
        Returns: actions in [-1,1], next_joints after tracking blend.
        """
        self.eval()
        B = prev_joints.size(0)
        dev = prev_joints.device
        time_feat = torch.zeros(B, self.time_dim, device=dev)  # single-step: use zeros or provide t externally
        prev_norm = (prev_joints - self.joint_mean) / self.joint_range
        feats = [prev_norm, z, time_feat]
        if self.obs_proj is not None and obs_t is not None:
            feats.append(self.obs_proj(obs_t))
        inp = torch.cat(feats, dim=-1).unsqueeze(1)  # [B,1,inp]

        # zero state for single step
        h0 = self.z_to_hidden(z).unsqueeze(0)
        c0 = self.z_to_cell(z).unsqueeze(0)
        out, _ = self.rnn(inp, (h0, c0))
        h_t = out[:, 0, :]
        action = self.policy_head(h_t)  # [-1,1]
        target = action * self.action_scale + self.default_dof_pos
        next_joints = prev_joints + self.track_alpha * (target - prev_joints)
        clamped = torch.max(torch.min(next_joints, self.joint_upper), self.joint_lower)
        next_joints = next_joints + (clamped - next_joints).detach()
        return action, next_joints

    def forward(
        self,
        z: torch.Tensor,                     # [B, Z]
        obs_seq: Optional[torch.Tensor]=None,# [B, T, obs_dim] or None
        q: Optional[torch.Tensor]=None,  # [B, T, DoF] or None
        dq: Optional[torch.Tensor]=None, # [B, T, DoF] or None
        object_override: Optional[torch.Tensor]=None, # [B, T, 3] optional
        tf_ratio: float = 1.0               # teacher forcing ratio (1.0 == full TF)
    ):
        """
        Returns:
          graph_x_mu:   [B, T, pos_dim]  normalized (for AE/NLL)
          joint_traj:   [B, T, DoF]      unnormalized
          actions_seq:  [B, T, DoF]      in [-1, 1]
          log_sigma_pos:[B, T, pos_dim]
        """
        B = z.size(0)
        dev = z.device

        # initial RNN state from z
        h = self.z_to_hidden(z).unsqueeze(0)
        c = self.z_to_cell(z).unsqueeze(0)

        # time features
        time_feats = self._time_features(self.seq_len, dev).unsqueeze(0).expand(B, -1, -1)

        # carry state (closed-loop)
        prev_free = self.default_dof_pos.unsqueeze(0).expand(B, -1)

        all_joints, all_actions, all_hidden, all_objects = [], [], [], []

        for t in range(self.seq_len):
            # choose input joints: TF uses t-1; free-roll uses carried prev
            use_teacher = self.training and (q is not None) and (torch.rand(()) < tf_ratio)
            if use_teacher:
                if t == 0:
                    # 🔧 expand default pose to batch
                    prev_for_input = self.default_dof_pos.unsqueeze(0).expand(B, -1)  # [B, DoF]
                else:
                    prev_for_input = q[:, t - 1, :]  # [B, DoF]
            else:
                prev_for_input = prev_free  # [B, DoF]
            prev_norm = (prev_for_input - self.joint_mean) / self.joint_range
            feats = [prev_norm, z, time_feats[:, t, :]]
            if (self.obs_proj is not None) and (obs_seq is not None):
                feats.append(self.obs_proj(obs_seq[:, t, :]))
            lstm_input = torch.cat(feats, dim=-1).unsqueeze(1)  # [B,1,inp]

            out, (h, c) = self.rnn(lstm_input, (h, c))
            h_t = out[:, 0, :]
            all_hidden.append(h_t)

            # policy action and target tracking
            action = self.policy_head(h_t)  # [-1,1]
            target = action * self.action_scale + self.default_dof_pos  # Go2Env style
            current_joints = prev_for_input + self.track_alpha * (target - prev_for_input)

            # straight-through clamp
            clamped = torch.max(torch.min(current_joints, self.joint_upper), self.joint_lower)
            current_for_input = current_joints + (clamped - current_joints).detach()

            # object
            if object_override is not None:
                current_object = object_override[:, t, :]
            elif self.predict_object:
                current_object = self.mlp_object(h_t)
            else:
                current_object = torch.zeros(B, 3, device=dev)

            all_actions.append(action)
            all_joints.append(current_joints)
            all_objects.append(current_object)

            # carry
            prev_free = current_for_input.detach()

        # stack
        actions_seq = torch.stack(all_actions, dim=1)       # [B,T,DoF]
        joint_traj  = torch.stack(all_joints,  dim=1)       # [B,T,DoF]
        H_seq       = torch.stack(all_hidden,  dim=1)       # [B,T,H]
        object_traj = torch.stack(all_objects, dim=1)       # [B,T,3]

        # FK to positions
        joint_flat  = joint_traj.reshape(B * self.seq_len, self.joint_dim).float()
        pos_flat    = self.fk_model(joint_flat)
        agent_traj  = pos_flat.view(B, self.seq_len, -1).float()

        # combine + normalize
        combined = torch.cat([agent_traj, object_traj], dim=-1)  # [B,T,pos_dim]
        combined_rs = combined.view(B, self.seq_len, self.num_links + 1, 3)
        combined_norm = (combined_rs - self.pos_mean) / self.pos_std
        graph_x_mu = combined_norm.view(B, self.seq_len, -1)

        # heteroscedastic sigma
        sigma_raw = self.var_head(H_seq)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        log_sigma_pos = torch.log(sigma)

        return graph_x_mu, joint_traj, actions_seq, log_sigma_pos
