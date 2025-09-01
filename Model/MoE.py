import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MoEActorCritic(nn.Module):
    """
    Weighted-Gaussian MoE with command head.
    - K frozen experts (rsl_rl PPO alg copies), each queried to produce μ_k(observations_with_relevant_commands_k)
    - Gating w = softmax(g(o_cmd)) with top-k sparsification (k=2) + stickiness (EMA) for sequencing
    - Command head c = sigmoid(h(o)) in [0,1] (env scales to ranges)
    - Final policy: N(mu_mix, diag(exp(2*log_std))) with mu_mix = sum_k w_k * μ_k
    """
    def __init__(self, obs_dim, act_dim, num_cmd, experts_algs, obs_normalizer,
                 cmd_lows, cmd_highs, hidden=256, topk=2, stickiness=0.85):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_cmd = num_cmd
        self.experts = experts_algs  # list of frozen PPO algs
        self.K = len(experts_algs)
        self.topk = topk
        self.stickiness = stickiness
        self.obs_norm = obs_normalizer  # RunningMeanStd; read mean/var live

        # Gating & value heads (PPO will update these)
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, self.K)
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Command head → [0,1]^num_cmd
        self.cmd_head = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, num_cmd)
        )

        # shared log-std for the collapsed Gaussian
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # per-env sticky weights (filled via reset_prev(num_envs, device))
        self.register_buffer("prev_w", None)

        # command ranges (for normalization injection)
        self.register_buffer("cmd_lows",  cmd_lows.clone())
        self.register_buffer("cmd_highs", cmd_highs.clone())

        # expert → command mask (union layout: 0–2 locomote, 3–5 bodypose, 6–9 limbvel, 10–12 contact, 13–15 hop)
        base_masks = []
        m = torch.zeros(num_cmd); m[:3]  = 1.0; base_masks.append(m.clone())   # locomote
        m = torch.zeros(num_cmd); m[3:6] = 1.0; base_masks.append(m.clone())   # bodypose
        m = torch.zeros(num_cmd); m[6:10]= 1.0; base_masks.append(m.clone())   # limbvel
        m = torch.zeros(num_cmd); m[10:13]=1.0; base_masks.append(m.clone())   # contact hold
        m = torch.zeros(num_cmd); m[13:16]=1.0; base_masks.append(m.clone())   # hop

        # repeat masks across K experts (K may be >5 if you loaded more checkpoints)
        self.cmd_masks = torch.stack([base_masks[i % len(base_masks)] for i in range(self.K)], dim=0)  # [K, C]
        self.register_buffer("cmd_masks_buf", self.cmd_masks)

        # will be filled on each act() for the rollout loop to set into env
        self.last_cmd01 = None     # [N, C] in [0,1]
        self.last_gate_w = None    # [N, K]

    def reset_prev(self, num_envs, device):
        self.prev_w = torch.full((num_envs, self.K), 1.0/self.K, device=device)

    # --- helpers ---
    def _topk_sticky_weights(self, logits):
        w = torch.softmax(logits, dim=-1)  # [N,K]
        if self.topk < self.K:
            v, i = torch.topk(w, self.topk, dim=-1)
            mask = torch.zeros_like(w).scatter(-1, i, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-8)
        if self.prev_w is None or self.prev_w.shape[0] != w.shape[0]:
            self.prev_w = torch.full_like(w, 1.0/self.K)
        w = self.stickiness * self.prev_w + (1.0 - self.stickiness) * w
        self.prev_w = w.detach()
        return w

    def _scale_cmd01_to_range(self, cmd01):  # cmd01: [N,C] in [0,1]
        return self.cmd_lows + cmd01 * (self.cmd_highs - self.cmd_lows)

    def _norm_cmd_full(self, cmd_full):  # cmd_full: [N,C] in env units
        # inject normalized commands into obs: (x - mean) / sqrt(var + eps)
        mean = self.obs_norm.mean[:self.num_cmd].to(cmd_full.device)
        var  = self.obs_norm.var[:self.num_cmd].to(cmd_full.device)
        return (cmd_full - mean) / torch.sqrt(var + self.obs_norm.epsilon)

    # --- core API used by PPO ---
    def act(self, obs_n, critic_obs_n):
        """
        obs_n, critic_obs_n: normalized observations from the runner's normalizer.
        Returns: action sampled (PPO wrapper will handle storage); we keep commands in self.last_cmd01 for the env.
        """
        N = obs_n.shape[0]

        # 1) Command synthesis (shared across experts)
        cmd_logits = self.cmd_head(obs_n)         # [N,C]
        cmd01 = torch.sigmoid(cmd_logits)         # [N,C] in [0,1]
        cmd_full = self._scale_cmd01_to_range(cmd01)     # env-range
        #cmd_norm = self._norm_cmd_full(cmd_full)         # normalized for experts/gate

        # 2) Build obs with shared command for gating & value
        obs_gate = obs_n.clone()
        obs_gate[:, :self.num_cmd] = cmd_full

        # 3) Gating (top-k + stickiness)
        logits = self.gate(obs_gate)              # [N,K]
        w = self._topk_sticky_weights(logits)     # [N,K]

        # 4) Query frozen experts with per-expert command slices
        mus = []
        for k, ex in enumerate(self.experts):
            # expert-specific normalized commands (mask others to zero)
            cmd_k_full = cmd_full * self.cmd_masks_buf[k].unsqueeze(0)      # zero non-relevant dims
            #cmd_k_norm = self._norm_cmd_full(cmd_k_full)

            obs_k = obs_n.clone()
            obs_k[:, :self.num_cmd] = cmd_k_full
            critic_obs_k = critic_obs_n.clone()
            critic_obs_k[:, :self.num_cmd] = cmd_k_full

            # use expert's act to get its μ_k proposal (deterministic enough; experts are frozen)
            # returns [N, A]; treat as μ_k
            mu_k = ex.act(obs_k, critic_obs_k)    # shape [N, A]
            mus.append(mu_k)
        mus = torch.stack(mus, dim=1)             # [N, K, A]

        # 5) Mixture mean and diagonal Gaussian
        mu_mix = (w.unsqueeze(-1) * mus).sum(dim=1)          # [N,A]
        std = self.log_std.exp().unsqueeze(0)                # [1,A]
        a = mu_mix + std * torch.randn_like(mu_mix)          # reparam sample
        # PPO will recompute logp/entropy during its internal calls to evaluate()

        # 6) Value
        V = self.value(obs_gate).squeeze(-1)      # [N]

        # bookkeeping for the rollout loop & debugging
        self.last_cmd01 = cmd01.detach()
        self.last_gate_w = w.detach()

        # rsl_rl PPO usually expects (action, logp, value, entropy, infos) downstream,
        # but the algorithm wrapper you use (`policy_alg.act`) handles storage; we only need to return action here.
        return a

    def evaluate_actions(self, obs_n, critic_obs_n, actions):
        """
        Must mirror act(): recompute mu_mix and produce log_prob, entropy, value for PPO update.
        """
        N = obs_n.shape[0]

        cmd01 = torch.sigmoid(self.cmd_head(obs_n))
        cmd_full = self._scale_cmd01_to_range(cmd01)
        #cmd_norm = self._norm_cmd_full(cmd_full)

        obs_gate = obs_n.clone()
        obs_gate[:, :self.num_cmd] = cmd_full
        logits = self.gate(obs_gate)
        w = torch.softmax(logits, dim=-1)
        if self.topk < self.K:
            v, i = torch.topk(w, self.topk, dim=-1)
            mask = torch.zeros_like(w).scatter(-1, i, 1.0)
            w = (w * mask) / (w * mask).sum(-1, keepdim=True).clamp_min(1e-8)

        mus = []
        for k, ex in enumerate(self.experts):
            cmd_k_full = cmd_full * self.cmd_masks_buf[k].unsqueeze(0)
            #cmd_k_norm = self._norm_cmd_full(cmd_k_full)
            obs_k = obs_n.clone(); obs_k[:, :self.num_cmd] = cmd_k_full
            critic_obs_k = critic_obs_n.clone(); critic_obs_k[:, :self.num_cmd] = cmd_k_full
            mu_k = ex.act(obs_k, critic_obs_k)      # [N,A]
            mus.append(mu_k)
        mus = torch.stack(mus, dim=1)               # [N,K,A]
        mu_mix = (w.unsqueeze(-1) * mus).sum(dim=1) # [N,A]

        std = self.log_std.exp().unsqueeze(0)       # [1,A]
        # log_prob and entropy of Independent Normal
        var = std.pow(2)
        log_prob = -0.5 * (((actions - mu_mix) ** 2) / var + 2 * self.log_std + math.log(2 * math.pi)).sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum().expand_as(log_prob)

        V = self.value(obs_gate).squeeze(-1)
        # infos (optional): gate weights, commands
        infos = {"gate_w": w.detach(), "cmd01": cmd01.detach()}
        return log_prob, V, entropy, infos
