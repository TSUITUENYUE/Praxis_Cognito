import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, GCNConv
import geoopt


class PoincareEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim):
        super(PoincareEncoder, self).__init__()

        # --- GNN and RNN architecture remains the same ---
        self.gnns = nn.ModuleList()
        self.gnns.append(GATConv(node_features, hidden_features, heads=1, concat=True))
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features, hidden_features, heads=1, concat=False))
        self.gnns.append(GATConv(hidden_features, hidden_features, heads=1, concat=False))
        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)

        # We will use the Poincaré Ball model. Curvature 'c' can be a tunable parameter.
        self.manifold = geoopt.PoincareBall(c=1.0).to('cuda')

        self.fc_mu = nn.Linear(rnn_hidden * 2, latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, latent_dim)

    def forward(self, x, edge_index):
        # The forward pass is identical to the VanillaEncoder's forward pass.
        # The ManifoldModule in self.fc_mu handles the projection automatically.
        bs, seq_len, num_nodes, _ = x.shape

        edge_indices = []
        for i in range(bs):
            offset = i * num_nodes
            edge_indices.append(edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)

        batch_idx = torch.arange(bs, device=x.device).repeat_interleave(num_nodes)

        graph_embs = []
        for t in range(seq_len):
            xt = x[:, t].reshape(bs * num_nodes, -1)
            for gnn in self.gnns:
                xt = F.relu(gnn(xt, batched_edge_index))
            emb = global_add_pool(xt, batch_idx,size=bs)
            graph_embs.append(emb)

        graph_embs = torch.stack(graph_embs, dim=1)
        _, (hn, cn) = self.rnn(graph_embs)
        temporal_feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        mu = self.fc_mu(temporal_feat)
        logvar = self.fc_logvar(temporal_feat)
        mu = self.manifold.projx(mu)
        return mu, logvar

class GMMEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim,num_components=32):
        super(GMMEncoder, self).__init__()

        self.gnns = nn.ModuleList()
        # Using GATConv with multi-head attention
        # The output of this layer will be hidden_features * heads
        self.gnns.append(GATConv(node_features, hidden_features, heads=1, concat=True))
        self.num_components = num_components
        self.latent_dim = latent_dim
        # Subsequent layers. The input features must match the output of the previous layer.
        # We use heads=1 and concat=False for the final GNN layer to get the desired output dimension.
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features, hidden_features, heads=1, concat=False))

        # The last layer maps back to the simple hidden_features dimension
        self.gnns.append(GATConv(hidden_features, hidden_features, heads=1, concat=False))

        # The RNN takes the GNN output as input
        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(rnn_hidden * 2, num_components * latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, num_components * latent_dim)
        self.fc_pi_logits = nn.Linear(rnn_hidden * 2, num_components)  # Logits for component probabilities

    def forward(self, x, edge_index):
        bs, seq_len, num_nodes, _ = x.shape

        edge_indices = []
        for i in range(bs):
            offset = i * num_nodes
            edge_indices.append(edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)

        batch_idx = torch.arange(bs, device=x.device).repeat_interleave(num_nodes)

        graph_embs = []
        for t in range(seq_len):
            # Reshape for GNN processing: [bs * num_nodes, features]
            xt = x[:, t].reshape(bs * num_nodes, -1)

            # Apply GNNs using the new batched_edge_index
            for gnn in self.gnns:
                xt = F.relu(gnn(xt, batched_edge_index))

            # Use global pooling
            emb = global_add_pool(xt, batch_idx,size=bs)
            graph_embs.append(emb)

        graph_embs = torch.stack(graph_embs, dim=1)

        _, (hn, cn) = self.rnn(graph_embs)

        # Correctly handle bidirectional RNN output
        temporal_feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        mu = self.fc_mu(temporal_feat).view(bs, self.num_components, self.latent_dim)
        logvar = self.fc_logvar(temporal_feat).view(bs, self.num_components, self.latent_dim)
        pi_logits = self.fc_pi_logits(temporal_feat)  # Shape: [bs, num_components]

        return mu, logvar, pi_logits

class VanillaEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim):
        super(VanillaEncoder, self).__init__()

        self.gnns = nn.ModuleList()
        # Using GATConv with multi-head attention
        # The output of this layer will be hidden_features * heads
        self.gnns.append(GATConv(node_features, hidden_features, heads=1, concat=True))

        # Subsequent layers. The input features must match the output of the previous layer.
        # We use heads=1 and concat=False for the final GNN layer to get the desired output dimension.
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features, hidden_features, heads=1, concat=False))

        # The last layer maps back to the simple hidden_features dimension
        self.gnns.append(GATConv(hidden_features, hidden_features, heads=1, concat=False))

        # The RNN takes the GNN output as input
        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(rnn_hidden * 2, latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, latent_dim)

    def forward(self, x, edge_index):
        bs, seq_len, num_nodes, _ = x.shape

        edge_indices = []
        for i in range(bs):
            offset = i * num_nodes
            edge_indices.append(edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)

        batch_idx = torch.arange(bs, device=x.device).repeat_interleave(num_nodes)

        graph_embs = []
        for t in range(seq_len):
            # Reshape for GNN processing: [bs * num_nodes, features]
            xt = x[:, t].reshape(bs * num_nodes, -1)

            # Apply GNNs using the new batched_edge_index
            for gnn in self.gnns:
                xt = F.relu(gnn(xt, batched_edge_index))

            # Use global pooling
            emb = global_add_pool(xt, batch_idx,size=bs)
            graph_embs.append(emb)

        graph_embs = torch.stack(graph_embs, dim=1)

        _, (hn, cn) = self.rnn(graph_embs)

        # Correctly handle bidirectional RNN output
        temporal_feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        mu = self.fc_mu(temporal_feat)
        logvar = self.fc_logvar(temporal_feat)

        return mu, logvar