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

@torch.no_grad()
def infer_contact_pointer_from_inputs(
    q: torch.Tensor,                 # [B,T,d]
    p: torch.Tensor,                 # [B,T,3]  WORLD
    w: torch.Tensor,                 # [B,T,4]  (w,x,y,z)
    u: torch.Tensor,                 # [B,T,3]  WORLD ball center
    valid_mask: torch.Tensor,        # [B,T] (1 valid, 0 pad)
    fk_model,                        # agent.fk_model, returns [*, L*3] in BASE
    link_bsphere_radius: torch.Tensor,  # [L]
    ball_radius: float = 0.05,
    sdf_tau: float = 0.02,
    smooth_kernel: int = 5,          # temporal smoothing on contact prob
    rise_temp: float = 4.0,          # softmax temperature for the rising-edge weights
    prob_thresh: float = 0.5         # for hard first-contact fallback
):
    """
    Returns:
      mu_star:     [B]   normalized time (0..1) of first contact onset
      sigma_star:  [B]   normalized width (softness) around onset
      contact_p:   [B,T] smoothed contact probability
      first_idx:   [B]   hard first-contact index (for debugging)
    """
    device = q.device
    B,T,d = q.shape
    BT = B*T

    # --- FK in BASE, then rotate & translate to WORLD ---
    link_pos_base = fk_model(q.reshape(BT, d)).view(B, T, -1, 3)      # [B,T,L,3]
    L = link_pos_base.shape[2]
    link_flat = link_pos_base.reshape(B*T*L, 3)
    quat_rep  = w.reshape(B*T, 4).repeat_interleave(L, dim=0)         # [B*T*L,4]

    # rotate BASE->WORLD; you can swap this to your own fast rot if preferred
    from genesis.utils.geom import transform_by_quat
    link_world_flat = transform_by_quat(link_flat, quat_rep)          # [B*T*L,3]
    link_world = link_world_flat.view(B, T, L, 3) + p.unsqueeze(2)    # [B,T,L,3]

    # --- min SDF(link-sphere, ball-sphere) over links ---
    radii = link_bsphere_radius.to(device=device, dtype=link_world.dtype).view(1,1,L)
    diff  = link_world - u.unsqueeze(2)                                # [B,T,L,3]
    dists = torch.linalg.norm(diff, dim=-1) - (radii + ball_radius)    # [B,T,L]
    sdf_min = dists.min(dim=2).values                                  # [B,T]

    # --- soft contact prob (match surrogate) + smoothing + padding ---
    contact_p = torch.sigmoid(-sdf_min / sdf_tau) * valid_mask         # [B,T]

    if smooth_kernel and smooth_kernel > 1:
        k = torch.ones(1, 1, smooth_kernel, device=device) / float(smooth_kernel)
        pad_l = smooth_kernel // 2
        pad_r = smooth_kernel - 1 - pad_l
        cp = F.pad(contact_p.unsqueeze(1), (pad_l, pad_r), mode="replicate")
        contact_p = F.conv1d(cp, k).squeeze(1) * valid_mask            # [B,T]

    # --- rising-edge weights (soft first-contact onset) ---
    # Only reward increases; ignore decreases.
    rise = F.relu(contact_p[:, 1:] - contact_p[:, :-1])                # [B,T-1]
    rise = F.pad(rise, (1,0))                                          # align to [B,T], t=0 has 0

    # normalize over valid frames (temperatured softmax over rise)
    # this focuses the pointer on the onset instead of the whole post window.
    rise_logits = rise * rise_temp + (valid_mask - 1.0) * 1e6          # -inf on pads
    w_rise = F.softmax(rise_logits, dim=1) * valid_mask                # [B,T]
    w_rise = w_rise / (w_rise.sum(dim=1, keepdim=True) + 1e-8)

    # expected index and variance under w_rise
    t_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # [1,T]
    t_exp = (w_rise * t_idx).sum(dim=1)                                       # [B]
    var   = (w_rise * (t_idx - t_exp.unsqueeze(1))**2).sum(dim=1)            # [B]

    lengths = valid_mask.sum(dim=1).clamp_min(1.0)     # [B]
    denom   = (lengths - 1.0).clamp_min(1.0)           # normalize by (L-1)

    mu_star    = (t_exp / denom).clamp(0.0, 1.0)       # in [0,1]
    sigma_star = (torch.sqrt(var) / denom).clamp(min=1e-3)

    # --- hard first-contact index (debug) ---
    over = (contact_p >= prob_thresh) * (valid_mask > 0)
    # first index where contact turns on
    first_idx = torch.full((B,), T-1, device=device, dtype=torch.long)
    any_on = over.any(dim=1)
    if any_on.any():
        # cumulative trick to find first True per row
        c = over.cumsum(dim=1)
        first_mask = (c == 1) & over
        first_pos = first_mask.float() * t_idx
        first_idx_found = first_pos.argmax(dim=1)  # argmax over zero/idx works with unique 1
        first_idx = torch.where(any_on, first_idx_found.long(), first_idx)

    return mu_star, sigma_star, contact_p, first_idx

def get_beta(epoch, total_epochs, strategy='cyclical', num_cycles=4, max_beta=1.0, warmup_epochs=20):
    if strategy == 'warmup':
        return 0.0 if epoch < warmup_epochs else max_beta * (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    elif strategy == 'cyclical':
        cycle_length = max(1, total_epochs // num_cycles)
        cycle_progress = (epoch % cycle_length) / cycle_length
        return max_beta * (1 / (1 + math.exp(-10 * (cycle_progress - 0.5))))
    else:
        return max_beta

def masked_quat_geodesic(pred_w, tgt_w, mask, eps=1e-8):
    # pred_w, tgt_w: [B,T,4] (wxyz, normalized)
    # mask: [B,T,1] or [B,T]
    # geodesic distance: 2*acos(|<q1,q2>|); we can use (1 - dot^2) as a smooth proxy
    pred = pred_w / (pred_w.norm(dim=-1, keepdim=True) + eps)
    tgt  = tgt_w  / (tgt_w.norm(dim=-1, keepdim=True) + eps)
    dot  = (pred * tgt).sum(dim=-1).abs().clamp(max=1.0)       # [B,T]
    # proxy loss (smooth, scale ~ angle^2 near 0)
    loss = 1.0 - dot**2
    if mask.dim() == 3: mask = mask.squeeze(-1)
    # average over unmasked
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom

