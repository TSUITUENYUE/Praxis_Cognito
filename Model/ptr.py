import torch
import torch.nn as nn
import torch.nn.functional as F

class AnalyticPointer(nn.Module):
    """
    Analytic (non-learned) Gaussian pointer computed from the FK graph + object node.

    Assumptions:
      - x[..., :3] are 3D positions (meters). If your x is normalized, pass pos_mean/pos_std to denormalize.
      - The last node is the object (set obj_index if different).
      - If link_r / ball_r are provided, uses sphere–sphere SDF; otherwise uses point–point distance.

    Returns per batch (B):
      mu_star in (0,1), sigma_star > 0, alpha_time [B, T] (Gaussian weights over time)
    """
    def __init__(
        self,
        obj_index: int = -1,
        pos_slice = slice(0, 3),
        link_r: torch.Tensor = None,     # [L] or None
        ball_r: float = 0.05,
        sdf_tau: float = 0.02,
        temp: float = 6.0,               # softmax temp for rising-edge
        smooth: int = 0,                 # temporal smoothing kernel (0 disables)
        sigma_floor: float = 1e-3,
        sigma_cap: float = 0.10,
        pos_mean: torch.Tensor = None,   # [3] or None
        pos_std: torch.Tensor = None,    # [3] or None
    ):
        super().__init__()
        self.obj_index   = obj_index
        self.pos_slice   = pos_slice
        self.ball_r      = float(ball_r)
        self.sdf_tau     = float(sdf_tau)
        self.temp        = float(temp)
        self.smooth      = int(smooth)
        self.sigma_floor = float(sigma_floor)
        self.sigma_cap   = float(sigma_cap)

        if link_r is not None:
            self.register_buffer("link_r", link_r.view(1, 1, -1))  # [1,1,L]
        else:
            self.link_r = None

        if pos_mean is not None and pos_std is not None:
            self.register_buffer("pos_mean", pos_mean.view(1, 1, 1, -1))
            self.register_buffer("pos_std",  pos_std.view(1, 1, 1, -1))
        else:
            self.pos_mean = None
            self.pos_std  = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x:    [B, T, N, 3] (positions expected in x[..., pos_slice])
        mask: [B, T, 1] (float {0,1})
        """
        B, T, N, _ = x.shape
        valid = mask.squeeze(-1)  # [B,T]

        # positions (denorm if needed)
        pos = x[..., self.pos_slice]  # [B,T,N,3]
        if (self.pos_mean is not None) and (self.pos_std is not None):
            pos = pos * self.pos_std + self.pos_mean

        # object & links
        obj = pos[:, :, self.obj_index, :]              # [B,T,3]
        links = pos[:, :, torch.arange(N) != (N + self.obj_index) % N, :]  # all except object
        L = links.shape[2]

        # min SDF over links
        diff = links - obj.unsqueeze(2)                # [B,T,L,3]
        dist = torch.linalg.norm(diff, dim=-1)         # [B,T,L]
        if self.link_r is not None:
            sdf = dist - (self.link_r.expand(B, T, L) + self.ball_r)
        else:
            sdf = dist  # point distance proxy
        sdf_min = sdf.min(dim=2).values                # [B,T]

        # soft contact prob and (optional) smoothing
        c = torch.sigmoid(-sdf_min / self.sdf_tau) * valid  # [B,T]
        if self.smooth > 1:
            k = torch.ones(1, 1, self.smooth, device=x.device) / float(self.smooth)
            pad_l = self.smooth // 2
            pad_r = self.smooth - 1 - pad_l
            cp = F.pad(c.unsqueeze(1), (pad_l, pad_r), mode="replicate")
            c = F.conv1d(cp, k).squeeze(1) * valid

        # rising edge distribution (first touch evidence)
        rise = F.relu(c[:, 1:] - c[:, :-1])
        rise = F.pad(rise, (1, 0))                     # [B,T]
        logits = rise * self.temp + (valid - 1.0) * 1e6
        w_rise = F.softmax(logits, dim=1) * valid
        w_rise = w_rise / (w_rise.sum(dim=1, keepdim=True) + 1e-8)

        # μ*, σ* in normalized time
        t = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0)  # [1,T]
        t_exp = (w_rise * t).sum(dim=1)                                         # [B]
        var_t = (w_rise * (t - t_exp.unsqueeze(1))**2).sum(dim=1)               # [B]
        lengths = valid.sum(dim=1).clamp_min(1.0)                                # [B]
        denom   = (lengths - 1.0).clamp_min(1.0)                                 # [B]
        mu_star    = (t_exp / denom).clamp(0.0, 1.0)
        sigma_star = (var_t.sqrt() / denom).clamp(self.sigma_floor, self.sigma_cap)

        # Convert (μ*,σ*) to Gaussian α_t over time (masked)
        t_norm = t / denom.unsqueeze(1)                                         # [B,T]
        gauss = torch.exp(-0.5 * ((t_norm - mu_star.unsqueeze(1)) / sigma_star.unsqueeze(1))**2) * valid
        alpha_time = gauss / (gauss.sum(dim=1, keepdim=True) + 1e-8)            # [B,T]

        return mu_star, sigma_star, alpha_time
