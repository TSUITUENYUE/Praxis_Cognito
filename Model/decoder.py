import torch
import torch.nn as nn
from .fk import FKModel


class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, object_dim, joint_dim, agent, hidden_dim,pos_mean,pos_std):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.rnn = nn.LSTMCell(latent_dim + hidden_dim, hidden_dim)
        self.pos_mean = pos_mean
        self.pos_std = pos_std
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
        self.initial_hidden = nn.Parameter(torch.zeros(hidden_dim))
        self.initial_cell = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, z):
        # z has shape [batch_size, latent_dim]

        # 1. Get the batch size from the input tensor z.
        batch_size = z.shape[0]

        # 2. Expand the initial hidden and cell states to match the batch size.
        h = self.initial_hidden.expand(batch_size, -1)  # Shape: [batch_size, hidden_dim]
        c = self.initial_cell.expand(batch_size, -1)  # Shape: [batch_size, hidden_dim]

        object_traj = []
        agent_traj = []
        joint_traj = []

        for t in range(self.seq_len):
            # Now z and h are both 2D, so concatenation works.
            # input_t shape: [batch_size, latent_dim + hidden_dim]
            input_t = torch.cat([z, h], dim=1)

            # The LSTMCell, Linear, and FKModel will now process the entire batch in parallel.
            h, c = self.rnn(input_t, (h, c))
            object_out = self.mlp_object(h)
            joint_out = self.mlp_agent(h)
            position_out = self.fk_model(joint_out).view(batch_size, -1)

            object_traj.append(object_out)
            agent_traj.append(position_out)
            joint_traj.append(joint_out)

        num_link = int(position_out.shape[1] / 3) + 1
        # Stack along the time dimension.
        # object_traj shape: [seq_len, batch_size, object_dim]
        # agent_traj shape: [seq_len, batch_size, position_dim]
        object_traj = torch.stack(object_traj, dim=0)
        agent_traj = torch.stack(agent_traj, dim=0)
        joint_traj = torch.stack(joint_traj, dim=0)
        # 3. (Best Practice) Permute to make batch the first dimension.
        # New shape: [batch_size, seq_len, feature_dim]
        combined_traj = torch.cat([agent_traj, object_traj], dim=-1)
        # Normalize combined_traj using z-score (reshape to apply per coord)
        #pos_mean = self.pos_mean.to(device='cuda')
        #pos_std = self.pos_std.to(device='cuda')
        combined_traj_reshaped = combined_traj.view(self.seq_len, batch_size, num_link, 3)
        combined_traj_norm = (combined_traj_reshaped - self.pos_mean) / self.pos_std
        combined_traj_norm = combined_traj_norm.view(self.seq_len, batch_size, -1)
        return combined_traj_norm.permute(1, 0, 2), joint_traj.permute(1, 0, 2)