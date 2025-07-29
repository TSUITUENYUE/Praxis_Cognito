import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
import geoopt


class PoincareEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim):
        super(PoincareEncoder, self).__init__()
        self.heads = 4  # Set number of attention heads

        # ✅ Optimized GNN structure with multi-head attention
        self.gnns = nn.ModuleList()
        self.gnns.append(GATConv(node_features, hidden_features, heads=self.heads, concat=True))
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=self.heads, concat=True))
        self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))

        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)
        self.manifold = geoopt.PoincareBall(c=1.0).to('cuda')
        self.fc_mu = nn.Linear(rnn_hidden * 2, latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, latent_dim)

    def forward(self, x, edge_index):
        # ✅ Fully vectorized forward pass
        bs, seq_len, num_nodes, _ = x.shape
        xt = x.view(bs * seq_len * num_nodes, -1)

        num_graphs = bs * seq_len
        batched_edge_index = edge_index.repeat(1, num_graphs)
        offset = torch.arange(num_graphs, device=x.device).repeat_interleave(edge_index.size(1)) * num_nodes
        batched_edge_index += offset

        batch_idx = torch.arange(num_graphs, device=x.device).repeat_interleave(num_nodes)

        for gnn in self.gnns:
            xt = F.relu(gnn(xt, batched_edge_index))

        graph_embs = global_add_pool(xt, batch_idx, size=num_graphs)
        graph_embs = graph_embs.view(bs, seq_len, -1)

        _, (hn, cn) = self.rnn(graph_embs)
        temporal_feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        # Poincaré-specific output
        mu = self.fc_mu(temporal_feat)
        logvar = self.fc_logvar(temporal_feat)
        mu = self.manifold.projx(mu) # Project mu onto the Poincaré ball
        return mu, logvar

class GMMEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim, num_components=32):
        super(GMMEncoder, self).__init__()
        self.heads = 4  # Set number of attention heads
        self.num_components = num_components
        self.latent_dim = latent_dim

        # ✅ Optimized GNN structure with multi-head attention
        self.gnns = nn.ModuleList()
        self.gnns.append(GATConv(node_features, hidden_features, heads=self.heads, concat=True))
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=self.heads, concat=True))
        self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))

        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(rnn_hidden * 2, num_components * latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, num_components * latent_dim)
        self.fc_pi_logits = nn.Linear(rnn_hidden * 2, num_components)

    def forward(self, x, edge_index):
        # ✅ Fully vectorized forward pass
        bs, seq_len, num_nodes, _ = x.shape
        xt = x.view(bs * seq_len * num_nodes, -1)

        num_graphs = bs * seq_len
        batched_edge_index = edge_index.repeat(1, num_graphs)
        offset = torch.arange(num_graphs, device=x.device).repeat_interleave(edge_index.size(1)) * num_nodes
        batched_edge_index += offset

        batch_idx = torch.arange(num_graphs, device=x.device).repeat_interleave(num_nodes)

        for gnn in self.gnns:
            xt = F.relu(gnn(xt, batched_edge_index))

        graph_embs = global_add_pool(xt, batch_idx, size=num_graphs)
        graph_embs = graph_embs.view(bs, seq_len, -1)

        _, (hn, cn) = self.rnn(graph_embs)
        temporal_feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        # GMM-specific output
        mu = self.fc_mu(temporal_feat).view(bs, self.num_components, self.latent_dim)
        logvar = self.fc_logvar(temporal_feat).view(bs, self.num_components, self.latent_dim)
        pi_logits = self.fc_pi_logits(temporal_feat)

        return mu, logvar, pi_logits

class VanillaEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim):
        super(VanillaEncoder, self).__init__()
        self.heads = 4 # Set number of attention heads

        # --- Using multi-head attention ---
        self.gnns = nn.ModuleList()
        # First layer: maps node_features to hidden_features * heads
        self.gnns.append(GATConv(node_features, hidden_features, heads=self.heads, concat=True))

        # Middle layers: input is hidden_features * heads
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=self.heads, concat=True))

        # Last layer: maps to hidden_features. The output is averaged across heads.
        self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))

        # RNN input dimension is now `hidden_features` from the last GNN layer
        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(rnn_hidden * 2, latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, latent_dim)

    def forward(self, x, edge_index):
        # Original shape: [bs, seq_len, num_nodes, features]
        bs, seq_len, num_nodes, _ = x.shape

        # 1. Reshape to process all timesteps at once
        # New shape: [bs * seq_len * num_nodes, features]
        xt = x.view(bs * seq_len * num_nodes, -1)

        # 2. Create a batched edge_index for all graphs (bs * seq_len)
        num_graphs = bs * seq_len
        # Repeat the base edge_index for each graph
        batched_edge_index = edge_index.repeat(1, num_graphs)
        # Create offsets to add to the node indices
        offset = torch.arange(num_graphs, device=x.device).repeat_interleave(edge_index.size(1)) * num_nodes
        batched_edge_index += offset

        # 3. Create a batch index for pooling
        batch_idx = torch.arange(num_graphs, device=x.device).repeat_interleave(num_nodes)

        # 4. Run GNNs on the single large batch
        for gnn in self.gnns:
            xt = F.relu(gnn(xt, batched_edge_index))

        # 5. Pool node embeddings to get a graph embedding for each timestep
        # Output shape: [bs * seq_len, hidden_features]
        graph_embs = global_add_pool(xt, batch_idx, size=num_graphs)

        # 6. Reshape for RNN processing
        # New shape: [bs, seq_len, hidden_features]
        graph_embs = graph_embs.view(bs, seq_len, -1)

        # --- The rest of the model remains the same ---
        _, (hn, cn) = self.rnn(graph_embs)
        temporal_feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        mu = self.fc_mu(temporal_feat)
        logvar = self.fc_logvar(temporal_feat)

        return mu, logvar