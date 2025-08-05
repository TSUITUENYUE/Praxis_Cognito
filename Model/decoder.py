import torch
import torch.nn as nn
from .fk import FKModel


class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, object_dim, joint_dim, agent, hidden_dim, pos_mean, pos_std):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pos_mean = pos_mean
        self.pos_std = pos_std

        # ✅ Use the full, optimized nn.LSTM layer instead of LSTMCell
        self.rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)

        self.mlp_object = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim)
        )
        self.mlp_agent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim)
        )
        self.fk_model = agent.fk_model
        # The initial hidden/cell states for LSTM require an extra dimension for layers
        self.initial_hidden = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.initial_cell = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, z):
        # z has shape [batch_size, latent_dim]
        batch_size = z.shape[0]

        # ✅ 1. Prepare inputs for the full LSTM layer
        # Expand initial states to match the batch size
        h0 = self.initial_hidden.expand(1, batch_size, self.hidden_dim).contiguous()
        c0 = self.initial_cell.expand(1, batch_size, self.hidden_dim).contiguous()

        # The same latent vector z is used as input for every timestep.
        # Expand z to shape: [batch_size, seq_len, latent_dim]
        lstm_input = z.unsqueeze(1).expand(-1, self.seq_len, -1)

        # ✅ 2. Run the entire sequence through the LSTM in one go
        # all_h shape: [batch_size, seq_len, hidden_dim]
        all_h, _ = self.rnn(lstm_input, (h0, c0))

        # ✅ 3. Process all hidden states in parallel (no loop needed)
        # Reshape for parallel MLP processing: [batch_size * seq_len, hidden_dim]
        all_h_flat = all_h.reshape(batch_size * self.seq_len, self.hidden_dim)

        object_out_flat = self.mlp_object(all_h_flat)
        joint_out_flat = self.mlp_agent(all_h_flat)
        position_out_flat = self.fk_model(joint_out_flat)

        # ✅ 4. Reshape outputs back to sequence format
        # object_traj shape: [batch_size, seq_len, object_dim]
        object_traj = object_out_flat.view(batch_size, self.seq_len, -1)
        agent_traj = position_out_flat.view(batch_size, self.seq_len, -1)
        joint_traj = joint_out_flat.view(batch_size, self.seq_len, -1)

        # --- Normalization logic remains the same, but no permute needed ---
        num_link = agent_traj.shape[2] // 3
        combined_traj = torch.cat([agent_traj, object_traj], dim=-1)
        combined_traj_reshaped = combined_traj.view(batch_size, self.seq_len, num_link + 1, 3)
        combined_traj_norm = (combined_traj_reshaped - self.pos_mean) / self.pos_std
        combined_traj_norm = combined_traj_norm.view(batch_size, self.seq_len, -1)
        graph_x = combined_traj_norm
        graph_x = combined_traj
        return graph_x, joint_traj