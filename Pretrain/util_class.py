import torch
import torch.nn as nn
class TrajDiscriminator(nn.Module):
    """Small MLP that scores post-contact *trajectory descriptors* (pooled features)."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1)  # logits
        )

    def forward(self, desc):  # desc: [B, D]
        return self.net(desc).squeeze(-1)  # [B]

class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, device='cuda'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon
    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
# ---------- Option Scheduler (semi-MDP layer) ----------
class OptionScheduler:
    def __init__(self, num_envs: int, cmd_dim: int, device, dtype):
        assert cmd_dim >= 16, "OptionScheduler expects 16 command dims"
        self.num_envs = num_envs
        self.cmd_dim = cmd_dim

        self.idx_base = torch.tensor([0,1,2,3,4,5], device=device)
        self.idx_swing = torch.tensor([6,7,8,9], device=device)
        self.idx_hold = torch.tensor([10,11,12], device=device)
        self.idx_hop  = torch.tensor([13,14,15], device=device)

        self.base_active = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.base_ttl = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.base_cmd = torch.zeros(num_envs, self.idx_base.numel(), device=device, dtype=dtype)

        self.swing_active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.swing_ttl = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.swing_cmd = torch.zeros(num_envs, self.idx_swing.numel(), device=device, dtype=dtype)

        self.hold_active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.hold_ttl = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.hold_cmd = torch.zeros(num_envs, self.idx_hold.numel(), device=device, dtype=dtype)

        self.hop_active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.hop_ttl = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.hop_cmd = torch.zeros(num_envs, self.idx_hop.numel(), device=device, dtype=dtype)

        self.rng = torch.Generator(device=device)
        self.base_ttl_range  = (50, 150)
        self.swing_ttl_range = (6, 15)
        self.hold_ttl_range  = (5, 12)
        self.hop_ttl_range   = (15, 30)

        self.trig_thr_swing = 0.15
        self.trig_thr_hold  = 0.15
        self.trig_thr_hop   = 0.15

        self._base_initialized = False
        self._resample_ttl(self.base_ttl, self.base_ttl_range)

    def reset(self):
        self.base_active.fill_(True)
        self.base_ttl.zero_()
        self.base_cmd.zero_()
        self.swing_active.zero_(); self.swing_ttl.zero_(); self.swing_cmd.zero_()
        self.hold_active.zero_();  self.hold_ttl.zero_();  self.hold_cmd.zero_()
        self.hop_active.zero_();   self.hop_ttl.zero_();   self.hop_cmd.zero_()
        self._base_initialized = False
        self._resample_ttl(self.base_ttl, self.base_ttl_range)

    def _resample_ttl(self, ttl_tensor: torch.Tensor, ttl_range):
        low, high = ttl_range
        ttl_tensor.copy_(torch.randint(low, high + 1, (self.num_envs,), generator=self.rng, device=ttl_tensor.device))

    @torch.no_grad()
    def step(self, raw_cmd01: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(raw_cmd01)

        # First step after reset: adopt base immediately
        if not self._base_initialized:
            self.base_cmd.copy_(raw_cmd01.index_select(1, self.idx_base))
            self._resample_ttl(self.base_ttl, self.base_ttl_range)
            self._base_initialized = True

        # BASE
        self.base_ttl.sub_(1)
        expired = self.base_ttl <= 0
        if torch.any(expired):
            self.base_cmd[expired] = raw_cmd01.index_select(1, self.idx_base)[expired]
            self._resample_ttl(self.base_ttl, self.base_ttl_range)
        out[:, self.idx_base] = self.base_cmd

        # SWING
        self.swing_ttl.sub_(self.swing_active.to(self.swing_ttl.dtype))
        ended = (self.swing_active & (self.swing_ttl <= 0))
        if torch.any(ended): self.swing_active[ended] = False
        inactive = ~self.swing_active
        if torch.any(inactive):
            swing_raw = raw_cmd01.index_select(1, self.idx_swing)
            trig = (swing_raw.pow(2).sum(dim=1).sqrt() > self.trig_thr_swing) & inactive
            if torch.any(trig):
                self.swing_cmd[trig] = swing_raw[trig]
                self._resample_ttl(self.swing_ttl, self.swing_ttl_range)
                self.swing_active[trig] = True
        if torch.any(self.swing_active):
            out[self.swing_active][:, self.idx_swing] = self.swing_cmd[self.swing_active]

        # HOLD
        self.hold_ttl.sub_(self.hold_active.to(self.hold_ttl.dtype))
        ended = (self.hold_active & (self.hold_ttl <= 0))
        if torch.any(ended): self.hold_active[ended] = False
        inactive = ~self.hold_active
        if torch.any(inactive):
            hold_raw = raw_cmd01.index_select(1, self.idx_hold)
            trig = (hold_raw.pow(2).sum(dim=1).sqrt() > self.trig_thr_hold) & inactive
            if torch.any(trig):
                self.hold_cmd[trig] = hold_raw[trig]
                self._resample_ttl(self.hold_ttl, self.hold_ttl_range)
                self.hold_active[trig] = True
        if torch.any(self.hold_active):
            out[self.hold_active][:, self.idx_hold] = self.hold_cmd[self.hold_active]

        # HOP
        self.hop_ttl.sub_(self.hop_active.to(self.hop_ttl.dtype))
        ended = (self.hop_active & (self.hop_ttl <= 0))
        if torch.any(ended): self.hop_active[ended] = False
        inactive = ~self.hop_active
        if torch.any(inactive):
            hop_raw = raw_cmd01.index_select(1, self.idx_hop)
            trig = (hop_raw.pow(2).sum(dim=1).sqrt() > self.trig_thr_hop) & inactive
            if torch.any(trig):
                self.hop_cmd[trig] = hop_raw[trig]
                self._resample_ttl(self.hop_ttl, self.hop_ttl_range)
                self.hop_active[trig] = True
        if torch.any(self.hop_active):
            out[self.hop_active][:, self.idx_hop] = self.hop_cmd[self.hop_active]

        return out