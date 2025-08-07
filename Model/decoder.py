import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, object_dim, joint_dim, agent, hidden_dim, pos_mean, pos_std):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.joint_dim = joint_dim
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.agent = agent

        self.fk_model = self.agent.fk_model
        self.pos_mean = pos_mean
        self.pos_std = pos_std

        lstm_input_dim = self.joint_dim + self.latent_dim
        self.rnn = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)

        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.z_to_cell = nn.Linear(latent_dim, hidden_dim)

        self.mlp_object = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.object_dim)
        )
        self.mlp_agent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),
            nn.Tanh()
        )

        self.start_token = nn.Parameter(torch.zeros(1, 1, self.joint_dim))

        device = 'cuda'
        joint_limits_lower = torch.tensor(self.agent.joint_limits_lower, device=device, dtype=torch.float32)
        joint_limits_upper = torch.tensor(self.agent.joint_limits_upper, device=device, dtype=torch.float32)

        self.joint_range = (joint_limits_upper - joint_limits_lower) / 2.0
        self.joint_mean = (joint_limits_upper + joint_limits_lower) / 2.0


    def forward(self, z, teacher_joints=None):
        batch_size = z.shape[0]
        device = z.device

        h = self.z_to_hidden(z).unsqueeze(0)
        c = self.z_to_cell(z).unsqueeze(0)

        # The first input token is always the start token, normalized to the [-1, 1] range
        start_token_norm = (self.start_token - self.joint_mean.view(1, 1, -1)) / self.joint_range.view(1, 1, -1)
        prev_joints_norm = start_token_norm.expand(batch_size, -1, -1)

        all_joint_outputs = []
        all_object_outputs = []

        for t in range(self.seq_len):
            lstm_input = torch.cat([prev_joints_norm, z.unsqueeze(1)], dim=-1)
            output_h, (h, c) = self.rnn(lstm_input, (h, c))

            # Predict raw joint angles in [-1, 1] range via Tanh
            raw_joints_norm = self.mlp_agent(output_h.squeeze(1))

            # Scale to the robot's actual joint limits for loss calculation
            current_joints = raw_joints_norm * self.joint_range + self.joint_mean

            current_object = self.mlp_object(output_h.squeeze(1))

            all_joint_outputs.append(current_joints)
            all_object_outputs.append(current_object)

            if self.training and teacher_joints is not None:
                # If training, use the ground truth from this step as the input for the next step.
                # We must re-normalize it back into the [-1, 1] range to match the model's output space.
                teacher_step_unnorm = teacher_joints[:, t, :]
                prev_joints_norm = ((teacher_step_unnorm - self.joint_mean) / self.joint_range).unsqueeze(1)
            else:
                # If evaluating, use the model's own (normalized) prediction for the next step.
                # .detach() prevents gradients from flowing back through time.
                prev_joints_norm = raw_joints_norm.unsqueeze(1).detach()

        joint_traj = torch.stack(all_joint_outputs, dim=1)
        object_traj = torch.stack(all_object_outputs, dim=1)

        # --- Post-processing (FK and Normalization) ---
        joint_traj_flat = joint_traj.reshape(batch_size * self.seq_len, self.joint_dim)
        position_out_flat = self.fk_model(joint_traj_flat)
        agent_traj = position_out_flat.view(batch_size, self.seq_len, -1)

        combined_traj = torch.cat([agent_traj, object_traj], dim=-1)
        num_link = agent_traj.shape[2] // 3
        combined_traj_reshaped = combined_traj.view(batch_size, self.seq_len, num_link + 1, 3)
        combined_traj_norm = (combined_traj_reshaped - self.pos_mean) / self.pos_std
        graph_x = combined_traj_norm.view(batch_size, self.seq_len, -1)
        graph_x = combined_traj
        return graph_x, joint_traj