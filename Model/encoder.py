import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
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
# ---- Vanilla (Gaussian) Encoder ----
class VanillaEncoder(nn.Module):
    def __init__(self, node_features, hidden_features, num_layers, rnn_hidden, latent_dim):
        super().__init__()
        self.heads = 4

        # --- spatial GNN stack (kept, GCN 版本) ---
        self.gnns = nn.ModuleList()
        self.gnns.append(GCNConv(node_features, hidden_features))
        for _ in range(num_layers - 2):
            self.gnns.append(GCNConv(hidden_features, hidden_features))
        self.gnns.append(GCNConv(hidden_features, hidden_features))

        self.pool = NodeAttentionPool(hidden_features)

        # --- temporal encoder: keep bidirectional LSTM, but we will use per-time outputs ---
        self.rnn = nn.LSTM(hidden_features, rnn_hidden, batch_first=True, bidirectional=True)

        # --- Gaussian pointer head over time (produces μ in (0,1), σ > 0) ---
        self.ptr_head = nn.Sequential(
            nn.Linear(2 * rnn_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # -> [mu_hat, log_sigma]
        )

        # --- latent heads ---
        self.fc_mu     = nn.Linear(2 * rnn_hidden, latent_dim)
        self.fc_logvar = nn.Linear(2 * rnn_hidden, latent_dim)

        nn.init.zeros_(self.fc_mu.weight);     nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight); nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x, edge_index, mask):
        """
        x:    [B, T, N, F]
        mask: [B, T, 1]  (float {0,1})
        returns:
            mu, logvar, extras (dict：alpha_time, mu_hat, sigma)
        """
        B, T, N, Fdim = x.shape
        device = x.device
        xt = x.view(B * T * N, Fdim)

        # --- repeat edges across B*T graphs ---
        num_graphs = B * T
        E = edge_index.size(1)
        batched_edge_index = edge_index.repeat(1, num_graphs)
        offset = (torch.arange(num_graphs, device=device).repeat_interleave(E) * N)
        batched_edge_index = batched_edge_index + offset

        batch_idx = torch.arange(num_graphs, device=device).repeat_interleave(N)

        # --- spatial GNN per (b,t), then node-attn pool -> [B,T,H] ---
        for gnn in self.gnns:
            xt = F.relu(gnn(xt, batched_edge_index))
        graph_embs = self.pool(xt, batch_idx, num_graphs)           # [B*T, H]
        graph_embs = graph_embs.view(B, T, -1)                      # [B, T, H]

        # --- temporal masking ---
        mask_t = mask  # [B,T,1]
        graph_embs = graph_embs * mask_t

        lengths = mask_t.squeeze(-1).sum(dim=1).clamp_min(1).long()  # [B]
        # pack -> LSTM -> unpack
        packed = pack_padded_sequence(graph_embs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # [B, T, 2H]

        # --------- Gaussian pointer over time ---------
        denom = lengths.clamp_min(1).unsqueeze(1).to(device).float()   # [B,1]
        masked_sum = (rnn_out * mask_t).sum(dim=1)                     # [B, 2H]
        clip_feat  = masked_sum / denom                                 # [B, 2H]

        ptr_params = self.ptr_head(clip_feat)     # [B, 2]
        mu_hat     = ptr_params[:, 0].sigmoid()   # in (0,1), normalize time
        log_sigma  = ptr_params[:, 1].clamp(-4, 4)
        sigma      = F.softplus(log_sigma) + 1e-4 # > 0


        t_idx   = torch.arange(T, device=device).float().unsqueeze(0).expand(B, T)  # [B,T]
        # normalize
        denom_t = (lengths - 1).clamp_min(1).unsqueeze(1).to(device).float()        # [B,1]
        t_norm  = t_idx / denom_t                                                   # [B,T], 0..1

        gauss = torch.exp(-0.5 * ((t_norm - mu_hat.unsqueeze(1)) / sigma.unsqueeze(1)) ** 2)  # [B,T]
        gauss = gauss * mask_t.squeeze(-1)                                          # [B,T]
        alpha_time = gauss / (gauss.sum(dim=1, keepdim=True) + 1e-8)                # [B,T]

        # Gaussian temporal_feat
        temporal_feat = torch.bmm(alpha_time.unsqueeze(1), rnn_out).squeeze(1)      # [B, 2H]

        # --- latent heads ---
        mu     = self.fc_mu(temporal_feat)       # [B, Dz]
        logvar = self.fc_logvar(temporal_feat)   # [B, Dz]

        extras = {'alpha_time': alpha_time, 'mu_hat': mu_hat, 'sigma': sigma}
        return mu, logvar, extras
