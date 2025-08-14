import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class Decoder(nn.Module):
    """
    Conditional policy decoder: π(a_t | q_t-1, q̇_t-1, obs_t, z, t)
    Maintains VAE compatibility while being simpler
    """

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
            obs_dim: int = 51,
            fps: int = 30 # Go2 config
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.joint_dim = joint_dim
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.agent = agent
        self.fk_model = self.agent.fk_model
        self.frame_rate = fps
        # Normalization buffers (for FK output only)
        self.register_buffer("pos_mean", pos_mean)
        self.register_buffer("pos_std", pos_std)

        # Joint limits and defaults
        device = pos_mean.device
        jl = torch.as_tensor(self.agent.joint_limits_lower, device=device, dtype=torch.float32)
        ju = torch.as_tensor(self.agent.joint_limits_upper, device=device, dtype=torch.float32)
        self.register_buffer("joint_lower", jl)
        self.register_buffer("joint_upper", ju)
        self.register_buffer("joint_range", (ju - jl) / 2.0)
        self.register_buffer("joint_mean", (ju + jl) / 2.0)

        init_angles = torch.as_tensor(self.agent.init_angles, device=device, dtype=torch.float32)
        self.register_buffer("default_dof_pos", init_angles)

        # Sizes
        self.num_links = len(self.fk_model.link_names)
        self.pos_dim = (self.num_links + 1) * 3

        # === Conditional policy network ===
        # Input: q_norm + q̇ + obs + z + time
        policy_input_dim = (
                2 * self.joint_dim +  # normalized joints for network + velocity
                self.obs_dim +  # environment observation
                self.latent_dim # trajectory intention
        )

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),
            nn.Tanh()
        )

        # Object head (for VAE)
        self.object_head = nn.Linear(hidden_dim, self.object_dim)

        # Variance head (for VAE loss)
        self.var_head = nn.Linear(hidden_dim, self.pos_dim)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -2.0)

        self.alpha_logits = nn.Parameter(torch.full((joint_dim,), math.log(0.3)))
        self.sigma_min = 0.05
        self.sigma_max = 0.5
        self.action_scale = 0.25  # From go2_env

    def forward(
            self,
            z: torch.Tensor,  # [B, latent_dim]
            obs_seq: Optional[torch.Tensor] = None,  # [B, T, obs_dim] - environment observations
            q: Optional[torch.Tensor] = None,  # [B, T, joint_dim] - teacher joints
            dq: Optional[torch.Tensor] = None,  # [B, T, joint_dim] - joint velocities
            object_override: Optional[torch.Tensor] = None,
            tf_ratio: float = 1.0
    ):
        """
        Forward pass with proper teacher forcing
        """
        B = z.size(0)
        dev = z.device

        # Initialize
        all_joints = []
        all_actions = []
        all_objects = []
        all_features = []

        # Starting state
        prev_joints = self.default_dof_pos.unsqueeze(0).expand(B, -1)  # [B, joint_dim]
        prev_dq = torch.zeros(B, self.joint_dim, device=dev)

        for t in range(self.seq_len):
            # Teacher forcing decision (per tf_ratio)
            use_teacher = self.training and (q is not None) and (torch.rand(1).item() < tf_ratio)

            if use_teacher:
                if t == 0:
                    input_joints = self.default_dof_pos.unsqueeze(0).expand(B, -1)
                    input_dq = torch.zeros(B, self.joint_dim, device=dev)
                else:
                    input_joints = q[:, t - 1, :]  # Previous teacher joints (NOT normalized)
                    input_dq = dq[:, t - 1, :] if dq is not None else torch.zeros_like(input_joints)
            else:
                input_joints = prev_joints  # Previous predicted (NOT normalized)
                input_dq = prev_dq

            # Get current observation
            current_obs = obs_seq[:, t, :] if obs_seq is not None else torch.zeros(B, self.obs_dim, device=dev)

            # Normalize ONLY for network input (not for FK)
            joints_norm = (input_joints - self.joint_mean) / self.joint_range

            # Build policy input: [q_norm, q̇, obs, z]
            policy_input = torch.cat([
                joints_norm,  # Normalized for network
                input_dq,  # Joint velocities
                current_obs,  # Environment observation
                z,  # Latent intention
            ], dim=-1)

            # Forward through policy
            features = self.feature_net(policy_input)
            all_features.append(features)

            # Predict action
            action = self.action_head(features)  # [-1, 1]

            # Convert to target joints
            '''TO DO'''
            '''Add similar control strategy as go2env'''
            '''How to address the physics in the envs???'''

            target_joints = action * self.action_scale + self.default_dof_pos


            alpha = torch.sigmoid(self.alpha_logits)  # (d,), in (0,1)
            current_joints = input_joints + alpha * (target_joints - input_joints)

            # Clamp to limits
            current_joints = torch.clamp(current_joints, self.joint_lower, self.joint_upper)

            # Compute velocity for next step
            if use_teacher:
                current_joints = q[:, t, :]
                current_dq = dq[:, t, :] if dq is not None else torch.zeros_like(current_joints)
            else:
                current_dq = (current_joints - prev_joints) * self.frame_rate

            # Object prediction
            if object_override is not None:
                current_object = object_override[:, t, :]
            else:
                current_object = self.object_head(features)

            # Store
            all_actions.append(action)
            all_joints.append(current_joints)
            all_objects.append(current_object)

            # Update for next step (only in free-roll mode)
            if not use_teacher:
                prev_joints = current_joints.detach()
                prev_dq = current_dq.detach()

        # Stack sequences
        actions_seq = torch.stack(all_actions, dim=1)
        joint_traj = torch.stack(all_joints, dim=1)  # RAW joints for FK
        object_traj = torch.stack(all_objects, dim=1)
        features_seq = torch.stack(all_features, dim=1)

        # FK with RAW joints (not normalized!)
        joint_flat = joint_traj.reshape(B * self.seq_len, self.joint_dim).float()
        pos_flat = self.fk_model(joint_flat)  # This needs raw joint angles
        agent_traj = pos_flat.view(B, self.seq_len, -1).float()

        # Combine positions
        combined_traj = torch.cat([agent_traj, object_traj], dim=-1)

        # Normalize ONLY the FK output for VAE loss
        combined_reshaped = combined_traj.view(B, self.seq_len, self.num_links + 1, 3)
        combined_norm = (combined_reshaped - self.pos_mean) / self.pos_std
        graph_x_mu = combined_norm.view(B, self.seq_len, -1)

        # Variance prediction
        sigma_raw = self.var_head(features_seq)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        log_sigma_pos = torch.log(sigma)

        return graph_x_mu, joint_traj, actions_seq, log_sigma_pos