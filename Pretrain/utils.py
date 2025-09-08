import torch
from torch_geometric.utils import dense_to_sparse
import math
import torch.nn.functional as F


def build_edge_index(fk_model, end_effector_indices, device):
    link_names = fk_model.link_names
    num_links = len(link_names)
    num_nodes = num_links + 1  # +1 for object
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    for joint_name, joint in fk_model.joint_map.items():
        if joint['parent'] in link_names and joint['child'] in link_names:
            p_idx = link_names.index(joint['parent'])
            c_idx = link_names.index(joint['child'])
            adj[p_idx, c_idx] = 1
            adj[c_idx, p_idx] = 1

    obj_index = num_links
    for ee_idx in end_effector_indices:
        adj[ee_idx, obj_index] = 1
        adj[obj_index, ee_idx] = 1

    edge_index_maybe_float, _ = dense_to_sparse(adj)
    return edge_index_maybe_float.long()


def gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF using erf; x can be any shape."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

@torch.no_grad()  # remove this decorator if you want gradients to flow into μ/σ via the masks
def build_soft_masks_from_sigma(
    mu_hat: torch.Tensor,      # [B], in (0,1)
    sigma: torch.Tensor,       # [B], >0 (pointer width)
    lengths: torch.Tensor,     # [B], valid lengths per sequence
    T: int,
    valid_mask: torch.Tensor,  # [B,T], 1 for valid, 0 for pad
    scale: float = 1.0,        # optional global scale for σ used in masks
    min_sigma: float = 1e-3,   # lower bound to avoid saturation
):
    """
    Build soft pre/post masks directly from the Gaussian pointer parameters.
    The boundary softness is σ (optionally scaled).
    Returns:
        t_norm: [B,T] normalized time in [0,1]
        m_pre:  [B,T] soft pre-contact mask (0..1), zero on pads
        m_post: [B,T] soft post-contact mask (0..1), zero on pads
        w_pre:  [B,T] normalized weights over pre frames (sum=1 per sample if any)
        w_post: [B,T] normalized weights over post frames (sum=1 per sample if any)
    """
    device = mu_hat.device
    B = mu_hat.shape[0]
    eps = 1e-8

    # Per-sample normalized timeline [0,1] that ignores pad tail
    t_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)  # [B,T]
    denom_t = (lengths - 1).clamp_min(1).unsqueeze(1).to(torch.float32)                   # [B,1]
    t_norm = t_idx / denom_t                                                               # [B,T]

    # Effective σ for the masks
    sigma_eff = (sigma * scale).clamp_min(min_sigma).unsqueeze(1)  # [B,1]

    # Gaussian-CDF step at μ: pre is 1 before μ, 0 after (soft)
    z = (mu_hat.unsqueeze(1) - t_norm) / sigma_eff                 # [B,T]
    m_pre = gaussian_cdf(z) * valid_mask                           # [B,T]
    m_post = (valid_mask - m_pre).clamp(min=0.0)                   # ensures pre+post = valid

    # Normalize to get per-segment weights (useful for weighted losses/pooling)
    w_pre = m_pre / (m_pre.sum(dim=1, keepdim=True) + eps)
    w_post = m_post / (m_post.sum(dim=1, keepdim=True) + eps)
    return t_norm, m_pre, m_post, w_pre, w_post
