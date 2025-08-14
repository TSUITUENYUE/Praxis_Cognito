import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---- Shared attention pooling module ----
class NodeAttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x, batch_idx, num_graphs):
        """
        x:         [N_total, F]
        batch_idx: [N_total] graph id for each node (0..num_graphs-1)
        returns:   [num_graphs, F]
        """
        logits = self.attn(x).squeeze(-1)               # [N_total]
        alpha = pyg_softmax(logits, batch_idx, num_nodes=num_graphs)   # segment softmax per graph
        x_weighted = x * alpha.unsqueeze(-1)            # [N_total, F]
        pooled = scatter_add(x_weighted, batch_idx, dim=0, dim_size=num_graphs)  # [G, F]
        return pooled

# ---- GMM Encoder ----
class GMMEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim, num_components=32):
        super().__init__()
        self.heads = 4
        self.num_components = num_components
        self.latent_dim = latent_dim

        # GNN stack
        self.gnns = nn.ModuleList()
        self.gnns.append(GATConv(node_features, hidden_features, heads=self.heads, concat=True))
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=self.heads, concat=True))
        self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))

        # Attention pooling over nodes (per (b,t) graph)
        self.pool = NodeAttentionPool(hidden_features)

        # Temporal encoder
        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)

        # Mixture heads
        self.fc_mu       = nn.Linear(rnn_hidden * 2, num_components * latent_dim)
        self.fc_logvar   = nn.Linear(rnn_hidden * 2, num_components * latent_dim)
        self.fc_pi_logits= nn.Linear(rnn_hidden * 2, num_components)

    def forward(self, x, edge_index, mask):
        """
        x:    [B, T, N, F]
        mask: [B, T, 1] float {0,1}
        """
        B, T, N, Fdim = x.shape
        xt = x.view(B * T * N, Fdim)  # nodes flattened

        # Build repeated edge_index for B*T graphs
        num_graphs = B * T
        E = edge_index.size(1)
        batched_edge_index = edge_index.repeat(1, num_graphs)
        offset = (torch.arange(num_graphs, device=x.device).repeat_interleave(E) * N)
        batched_edge_index = batched_edge_index + offset

        # Batch index per node
        batch_idx = torch.arange(num_graphs, device=x.device).repeat_interleave(N)  # [B*T*N]

        # GNN stack
        for gnn in self.gnns:
            xt = F.relu(gnn(xt, batched_edge_index))

        # Attention pool to per-(b,t) graph embeddings: [B*T, H]
        graph_embs = self.pool(xt, batch_idx, num_graphs)  # [B*T, H]
        graph_embs = graph_embs.view(B, T, -1)             # [B, T, H]

        # Apply temporal mask before RNN (broadcast over feature dim)
        graph_embs = graph_embs * mask  # [B,T,1] broadcasts to [B,T,H]

        # Packed LSTM so padded steps don't affect the latent
        lengths = mask.squeeze(-1).sum(dim=1).clamp_min(1).long()  # [B]
        packed = pack_padded_sequence(graph_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.rnn(packed)
        # hn: [2, B, rnn_hidden] for bidirectional, last layer only

        temporal_feat = torch.cat((hn[-2], hn[-1]), dim=1)  # [B, 2*rnn_hidden]

        mu       = self.fc_mu(temporal_feat).view(B, self.num_components, self.latent_dim)
        logvar   = self.fc_logvar(temporal_feat).view(B, self.num_components, self.latent_dim)
        pi_logits= self.fc_pi_logits(temporal_feat)  # [B, K]
        return mu, logvar, pi_logits

# ---- Vanilla (Gaussian) Encoder ----
class VanillaEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim):
        super().__init__()
        self.heads = 4

        self.gnns = nn.ModuleList()
        self.gnns.append(GATConv(node_features, hidden_features, heads=self.heads, concat=True))
        for _ in range(num_layers - 2):
            self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=self.heads, concat=True))
        self.gnns.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))

        self.pool = NodeAttentionPool(hidden_features)

        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(rnn_hidden * 2, latent_dim)
        self.fc_logvar = nn.Linear(rnn_hidden * 2, latent_dim)

        nn.init.zeros_(self.fc_mu.weight);     nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight); nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x, edge_index, mask):
        """
        x:    [B, T, N, F]
        mask: [B, T, 1]
        """
        B, T, N, Fdim = x.shape
        xt = x.view(B * T * N, Fdim)

        num_graphs = B * T
        E = edge_index.size(1)
        batched_edge_index = edge_index.repeat(1, num_graphs)
        offset = (torch.arange(num_graphs, device=x.device).repeat_interleave(E) * N)
        batched_edge_index = batched_edge_index + offset

        batch_idx = torch.arange(num_graphs, device=x.device).repeat_interleave(N)

        for gnn in self.gnns:
            xt = F.relu(gnn(xt, batched_edge_index))

        graph_embs = self.pool(xt, batch_idx, num_graphs)  # [B*T, H]
        graph_embs = graph_embs.view(B, T, -1)

        # temporal masking
        #print(graph_embs.shape, mask.shape)
        graph_embs = graph_embs * mask

        lengths = mask.squeeze(-1).sum(dim=1).clamp_min(1).long()
        packed = pack_padded_sequence(graph_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.rnn(packed)

        temporal_feat = torch.cat((hn[-2], hn[-1]), dim=1)  # [B, 2*rnn_hidden]
        mu = self.fc_mu(temporal_feat)
        logvar = self.fc_logvar(temporal_feat)
        return mu, logvar
