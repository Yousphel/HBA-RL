"""
Business-Aware Hierarchical RL Agent — Per-Node PPO
=====================================================
Revision 1 additions (supervisor feedback):
  * deterministic=True on both select_* methods: macro takes the argmax of each head,
    micro takes sigmoid(mean). Used for ALL evaluation (D8).
  * zero_subgoal=True: the micro-controllers receive g = 0 instead of the macro
    probabilities (ablation ABL-3, issue A17).
  * hparams() returns every hyper-parameter for Table M1 (D3); n_parameters() the sizes.
  * default entropy_coef_min is 0.001 (was 0.01 in the class, overridden to 0.001 by the script).

Evolved from Stable v3. Changes vs v3:

  Architecture (matches BA-HRL diagram exactly):
    1. HighLevelPolicy gains routing_head  → macro routes job to specific node
    2. Single LowLevelPolicy → nn.ModuleList of 5 LocalResourceAllocators
    3. high_action['tensor'] (3-int) → 12-dim softmax subgoal for richer micro conditioning
    4. NodeBuffer per node — each micro policy learns from its own experience

  Stability:
    5. target_kl: 0.01 → 0.05  (was causing false early-stop after epoch 1)

  Preserved from v3 (unchanged):
    - RunningMeanStd reward normalisation (Welford)
    - GAE (λ=0.95)
    - PPO clip (eps=0.1), value loss clipping
    - Entropy annealing (0.05 → 0.001)
    - Orthogonal init + LayerNorm
    - Cosine LR decay
    - Best model tracking + restore_best()
    - save_models / load_models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from seed_utils import set_global_seed


# ==================== Reward Normalizer ====================

class RunningMeanStd:
    """Online reward normalisation using Welford's algorithm."""
    def __init__(self, epsilon=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x):
        x           = np.asarray(x, dtype=float)
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = len(x)
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot_count
        M2          = (self.var * self.count
                       + batch_var * batch_count
                       + delta**2 * self.count * batch_count / tot_count)
        self.var    = M2 / tot_count
        self.count  = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ==================== Neural Networks ====================

class HighLevelPolicy(nn.Module):
    """
    Macro policy — strategic decisions:
      priority_head : which workload class to favour   (4 options)
      resource_head : resource allocation strategy     (4 options)
      sla_head      : SLA aggressiveness               (4 options)
      routing_head  : which NODE to route the job to   (num_nodes options)
      value_head    : critic

    The 3 softmax probability vectors (12-dim total) form the subgoal
    passed to the micro policy for rich conditioning.
    """

    def __init__(self, state_dim: int, subgoal_dim: int = 4,
                 num_nodes: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.num_nodes = num_nodes
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),    nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.priority_head = nn.Linear(hidden_dim // 2, subgoal_dim)
        self.resource_head = nn.Linear(hidden_dim // 2, subgoal_dim)
        self.sla_head      = nn.Linear(hidden_dim // 2, subgoal_dim)
        self.routing_head  = nn.Linear(hidden_dim // 2, num_nodes)   # NEW: routes to node
        self.value_head    = nn.Linear(hidden_dim // 2, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        for head in [self.priority_head, self.resource_head,
                     self.sla_head, self.routing_head, self.value_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)

    def forward(self, state):
        f = self.net(state)
        return {
            'priority': F.softmax(self.priority_head(f), dim=-1),
            'resource': F.softmax(self.resource_head(f), dim=-1),
            'sla':      F.softmax(self.sla_head(f),      dim=-1),
            'routing':  F.softmax(self.routing_head(f),  dim=-1),   # (B, num_nodes)
            'value':    self.value_head(f),
        }


class LocalResourceAllocator(nn.Module):
    """
    Micro policy — Local Resource Allocator for ONE node.
    Input : (state [50-dim] ++ subgoal [12-dim]) → 62-dim
    Output: [workload_idx ∈ [0,1], alloc ∈ [0,1]]  (2-dim continuous)

    Each node has its own instance — weights are NOT shared.
    Conditioned on macro subgoal (concatenated softmax probs from
    priority/resource/sla heads) for goal-directed scheduling.
    """
    SUBGOAL_DIM = 12   # 3 heads × 4 probs each

    def __init__(self, state_dim: int = 50, action_dim: int = 2,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.SUBGOAL_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std     = nn.Parameter(torch.zeros(action_dim))
        self.value_head  = nn.Linear(hidden_dim // 2, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def forward(self, state, subgoal):
        f    = self.net(torch.cat([state, subgoal], dim=-1))
        mean = self.action_mean(f)
        std  = torch.exp(self.log_std.clamp(-3, 1)).expand_as(mean)
        return mean, std, self.value_head(f)

    def sample_action(self, state, subgoal):
        mean, std, value = self.forward(state, subgoal)
        dist     = torch.distributions.Normal(mean, std)
        raw      = dist.sample()
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        return torch.sigmoid(raw), raw, log_prob, value

    def evaluate_action(self, state, subgoal, raw):
        mean, std, value = self.forward(state, subgoal)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        entropy  = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, value


# ==================== Per-Node Memory Buffer ====================

class NodeBuffer:
    """Rollout buffer for one node's micro policy."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.states    = []
        self.subgoals  = []
        self.raws      = []
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.dones     = []

    def add(self, state, subgoal, raw, log_prob, value, reward, done):
        self.states.append(state)
        self.subgoals.append(subgoal)
        self.raws.append(raw)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


# ==================== HRL Agent ====================

class HierarchicalRLAgent:
    """
    Business-Aware HRL Agent — Per-Node PPO.

    Macro (HighLevelPolicy — one shared):
      Prioritises + routes job to a specific node.
      Outputs: priority/resource/sla subgoal + routing decision.

    Micro (LocalResourceAllocator × num_nodes — one per node):
      Each node has its own PPO policy.
      Learns workload selection + allocation for its own node only.
      Conditioned on macro subgoal (12-dim softmax probs).

    Action assembly for env.step():
      env_action[0] = micro output[0]          (workload selection)
      env_action[1] = (node_idx + 0.5) / 5     (node — from macro routing)
      env_action[2] = micro output[1]          (allocation)
    """

    def __init__(self,
                 state_dim:          int   = 50,
                 num_nodes:          int   = 5,
                 high_level_lr:      float = 3e-4,
                 low_level_lr:       float = 3e-4,
                 gamma:              float = 0.99,
                 gae_lambda:         float = 0.95,
                 eps_clip:           float = 0.1,
                 k_epochs:           int   = 4,
                 entropy_coef:       float = 0.05,
                 entropy_coef_min:   float = 0.001,
                 value_loss_coef:    float = 0.5,
                 max_grad_norm:      float = 0.5,
                 target_kl:          float = 0.05,   # raised from 0.01 — was false-stopping
                 normalize_rewards:  bool  = True,
                 total_episodes:     int   = 2000,
                 seed:               int   = 42,
                 zero_subgoal:       bool  = False):

        self.zero_subgoal = zero_subgoal
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
        self.seed = seed

        self.num_nodes        = num_nodes
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.eps_clip         = eps_clip
        self.k_epochs         = k_epochs
        self.entropy_coef     = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_max = entropy_coef
        self.value_loss_coef  = value_loss_coef
        self.max_grad_norm    = max_grad_norm
        self.target_kl        = target_kl
        self.normalize_rewards = normalize_rewards
        self.total_episodes   = total_episodes
        self.update_count     = 0
        self._reward_window   = []   # smoothed window for best-model tracking

        # ── Macro policy (one shared) ────────────────────────────────
        self.high_level_policy = HighLevelPolicy(state_dim, num_nodes=num_nodes)
        self.high_opt = optim.Adam(
            self.high_level_policy.parameters(), lr=high_level_lr, eps=1e-5)
        self.high_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.high_opt, T_max=total_episodes // 10, eta_min=1e-5)

        # ── Micro policies (one per node) ────────────────────────────
        self.micro_policies = nn.ModuleList([
            LocalResourceAllocator(state_dim) for _ in range(num_nodes)
        ])
        self.micro_opts = [
            optim.Adam(p.parameters(), lr=low_level_lr, eps=1e-5)
            for p in self.micro_policies
        ]
        self.micro_schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_episodes // 10, eta_min=1e-5)
            for opt in self.micro_opts
        ]

        # ── Memory ───────────────────────────────────────────────────
        self._reset_macro_memory()
        self.node_buffers = [NodeBuffer() for _ in range(num_nodes)]

        # ── Reward normaliser ────────────────────────────────────────
        self.reward_rms = RunningMeanStd()

        # ── Best model tracking ──────────────────────────────────────
        self.best_reward       = -np.inf
        self.best_high_state   = None
        self.best_micro_states = [None] * num_nodes

        # ── Routing stats (logged every 50 eps) ─────────────────────
        self.routing_counts = np.zeros(num_nodes)

    # ------------------------------------------------------------------
    def _reset_macro_memory(self):
        self.macro_memory = {
            'states':      [],
            'high_actions': [],   # (T, 4): [p_idx, r_idx, s_idx, node_idx]
            'log_probs':   [],
            'values':      [],
            'rewards':     [],
            'dones':       [],
        }

    # ------------------------------------------------------------------
    def _anneal_entropy(self):
        progress = min(self.update_count / max(self.total_episodes // 10, 1), 1.0)
        self.entropy_coef = (self.entropy_coef_max
                             - (self.entropy_coef_max - self.entropy_coef_min) * progress)

    # ------------------------------------------------------------------
    def select_high_level_action(self, state: np.ndarray, deterministic: bool = False) -> Dict:
        """
        Macro decision: prioritise + route.
        Returns node_idx (routing) + 12-dim subgoal for micro.
        deterministic=True takes the argmax of every head (evaluation).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            out = self.high_level_policy(state_t)

        p_dist = torch.distributions.Categorical(out['priority'])
        r_dist = torch.distributions.Categorical(out['resource'])
        s_dist = torch.distributions.Categorical(out['sla'])
        n_dist = torch.distributions.Categorical(out['routing'])

        if deterministic:
            p_a, r_a, s_a, n_a = (out['priority'].argmax(-1), out['resource'].argmax(-1),
                                  out['sla'].argmax(-1), out['routing'].argmax(-1))
        else:
            p_a, r_a, s_a, n_a = (p_dist.sample(), r_dist.sample(),
                                   s_dist.sample(), n_dist.sample())

        log_prob = (p_dist.log_prob(p_a) + r_dist.log_prob(r_a) +
                    s_dist.log_prob(s_a) + n_dist.log_prob(n_a))

        # 12-dim subgoal: concatenated softmax probs from 3 strategic heads
        subgoal = torch.cat([
            out['priority'].squeeze(0),
            out['resource'].squeeze(0),
            out['sla'].squeeze(0),
        ])  # (12,)
        if self.zero_subgoal:                      # ABL-3: the micro-controllers get no information from g
            subgoal = torch.zeros_like(subgoal)

        if not deterministic:                      # training-time routing statistics only
            self.routing_counts[n_a.item()] += 1

        return {
            'node_idx':  n_a.item(),
            'p_idx':     p_a.item(),
            'r_idx':     r_a.item(),
            's_idx':     s_a.item(),
            'indices':   torch.tensor(
                [p_a.item(), r_a.item(), s_a.item(), n_a.item()],
                dtype=torch.float32),
            'subgoal':   subgoal,
            'log_prob':  log_prob,
            'value':     out['value'],
        }

    # ------------------------------------------------------------------
    def select_low_level_action(self, state: np.ndarray, high_action: Dict, deterministic: bool = False):
        """
        Micro decision for the routed node.
        Returns 2-dim action [workload_idx, alloc].
        Training script assembles final 3-dim env action.
        """
        node_idx  = high_action['node_idx']
        state_t   = torch.FloatTensor(state).unsqueeze(0)
        subgoal_t = high_action['subgoal'].unsqueeze(0)

        with torch.no_grad():
            if deterministic:                       # mean action, no sampling noise
                mean, _, value = self.micro_policies[node_idx](state_t, subgoal_t)
                action, raw = torch.sigmoid(mean), mean
                log_prob = torch.zeros(1, 1)
            else:
                action, raw, log_prob, value = \
                    self.micro_policies[node_idx].sample_action(state_t, subgoal_t)

        return (action.squeeze(0).numpy(),
                raw.squeeze(0),
                log_prob.squeeze(),
                value.squeeze())

    # ------------------------------------------------------------------
    def store_transition(self, state, high_action, low_action_raw,
                         low_log_prob, low_value, reward, done):
        """Route experience to correct node buffer + macro buffer."""
        node_idx = high_action['node_idx']

        # Macro buffer
        self.macro_memory['states'].append(state)
        self.macro_memory['high_actions'].append(high_action['indices'])
        self.macro_memory['log_probs'].append(high_action['log_prob'])
        self.macro_memory['values'].append(high_action['value'].squeeze())
        self.macro_memory['rewards'].append(reward)
        self.macro_memory['dones'].append(done)

        # Node-specific micro buffer
        self.node_buffers[node_idx].add(
            state,
            high_action['subgoal'],
            low_action_raw,
            low_log_prob,
            low_value,
            reward,
            done,
        )

    # ------------------------------------------------------------------
    def _compute_gae(self, rewards, values, dones, next_value=0.0):
        advantages, gae = [], 0.0
        vals_ext = values + [torch.tensor(next_value)]
        for i in reversed(range(len(rewards))):
            delta = (rewards[i]
                     + self.gamma * vals_ext[i+1].item() * (1 - dones[i])
                     - vals_ext[i].item())
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns    = advantages + torch.stack(values)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ------------------------------------------------------------------
    def _micro_ppo_update(self, policy, optimizer, states, subgoals,
                          raws, old_lps, returns, advantages):
        """Shared PPO update logic — used for each micro policy."""
        for _ in range(self.k_epochs):
            new_lp, entropy, new_val = policy.evaluate_action(states, subgoals, raws)
            new_lp  = new_lp.reshape(-1)
            new_val = new_val.reshape(-1)
            entropy = entropy.reshape(-1)

            ratio     = torch.exp(new_lp - old_lps)
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            old_v      = returns.detach()
            val_clip   = old_v + torch.clamp(new_val - old_v, -self.eps_clip, self.eps_clip)
            value_loss = torch.max(F.mse_loss(new_val, returns),
                                   F.mse_loss(val_clip, returns))

            loss = (-torch.min(surr1, surr2).mean()
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef    * entropy.mean())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            optimizer.step()

    # ------------------------------------------------------------------
    def update_policies(self, episode_reward: float = None):
        if not self.macro_memory['rewards']:
            return

        self.update_count += 1
        self._anneal_entropy()

        rewards = list(self.macro_memory['rewards'])
        if self.normalize_rewards:
            self.reward_rms.update(rewards)
            rewards = [self.reward_rms.normalize(r) for r in rewards]

        # ── Macro PPO update ─────────────────────────────────────────
        states    = torch.FloatTensor(np.array(self.macro_memory['states']))
        hi_acts   = torch.stack(self.macro_memory['high_actions'])

        hi_ret, hi_adv = self._compute_gae(
            rewards, self.macro_memory['values'], self.macro_memory['dones'])

        # Compute old routing log-probs separately — dedicated signal for routing head
        with torch.no_grad():
            out0           = self.high_level_policy(states)
            old_routing_lp = torch.distributions.Categorical(
                out0['routing']).log_prob(hi_acts[:, 3].long())
            old_strategic_lp = (
                torch.distributions.Categorical(out0['priority']).log_prob(hi_acts[:, 0].long()) +
                torch.distributions.Categorical(out0['resource']).log_prob(hi_acts[:, 1].long()) +
                torch.distributions.Categorical(out0['sla']).log_prob(hi_acts[:, 2].long())
            )

        for _ in range(self.k_epochs):
            out = self.high_level_policy(states)

            p_dist = torch.distributions.Categorical(out['priority'])
            r_dist = torch.distributions.Categorical(out['resource'])
            s_dist = torch.distributions.Categorical(out['sla'])
            n_dist = torch.distributions.Categorical(out['routing'])

            # ── Strategic heads (priority/resource/sla) ───────────────
            strategic_lp      = (p_dist.log_prob(hi_acts[:, 0].long()) +
                                 r_dist.log_prob(hi_acts[:, 1].long()) +
                                 s_dist.log_prob(hi_acts[:, 2].long()))
            entropy_strategic = (p_dist.entropy() + r_dist.entropy() + s_dist.entropy()).mean()

            ratio_s   = torch.exp(strategic_lp - old_strategic_lp)
            approx_kl = ((ratio_s - 1) - torch.log(ratio_s)).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break

            surr1_s = ratio_s * hi_adv
            surr2_s = torch.clamp(ratio_s, 1 - self.eps_clip, 1 + self.eps_clip) * hi_adv

            # ── Routing head — dedicated loss, weighted 2x ─────────────
            routing_lp    = n_dist.log_prob(hi_acts[:, 3].long())
            entropy_route = n_dist.entropy().mean()
            ratio_r       = torch.exp(routing_lp - old_routing_lp)
            surr1_r = ratio_r * hi_adv
            surr2_r = torch.clamp(ratio_r, 1 - self.eps_clip, 1 + self.eps_clip) * hi_adv

            # ── Value loss ─────────────────────────────────────────────
            new_val  = out['value'].reshape(-1)
            old_v    = hi_ret.detach()
            val_clip = old_v + torch.clamp(new_val - old_v, -self.eps_clip, self.eps_clip)
            vf_loss  = torch.max(F.mse_loss(new_val, hi_ret),
                                 F.mse_loss(val_clip, hi_ret))

            # Routing weighted 2x so it gets dedicated gradient signal
            loss = (-torch.min(surr1_s, surr2_s).mean()
                    - 2.0 * torch.min(surr1_r, surr2_r).mean()
                    + self.value_loss_coef * vf_loss
                    - self.entropy_coef * (entropy_strategic + entropy_route))

            self.high_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), self.max_grad_norm)
            self.high_opt.step()

        self.high_scheduler.step()

        # ── Per-node micro PPO updates ────────────────────────────────
        for node_idx, (policy, opt, sched, buf) in enumerate(zip(
                self.micro_policies, self.micro_opts,
                self.micro_schedulers, self.node_buffers)):

            if len(buf) == 0:
                continue   # node received no jobs this batch

            node_rewards = list(buf.rewards)
            if self.normalize_rewards:
                node_rewards = [self.reward_rms.normalize(r) for r in node_rewards]

            node_states   = torch.FloatTensor(np.array(buf.states))
            node_subgoals = torch.stack(buf.subgoals)
            node_raws     = torch.stack(buf.raws)
            old_lps       = torch.stack(buf.log_probs).detach().reshape(-1)

            node_ret, node_adv = self._compute_gae(
                node_rewards, buf.values, buf.dones)

            self._micro_ppo_update(
                policy, opt,
                node_states, node_subgoals, node_raws,
                old_lps, node_ret, node_adv)
            sched.step()
            buf.reset()

        # ── Best model tracking — smoothed window, not single episode ────
        # Single-episode peak is noisy; track 10-episode moving average
        # to select the checkpoint that generalises best to evaluation.
        self._reward_window.append(episode_reward if episode_reward is not None else 0.0)
        if len(self._reward_window) > 10:
            self._reward_window.pop(0)
        smoothed = sum(self._reward_window) / len(self._reward_window)

        if smoothed > self.best_reward:
            self.best_reward = smoothed
            self.best_high_state = {
                k: v.cpu().clone()
                for k, v in self.high_level_policy.state_dict().items()}
            self.best_micro_states = [
                {k: v.cpu().clone() for k, v in p.state_dict().items()}
                for p in self.micro_policies]

        self._reset_macro_memory()

    # ------------------------------------------------------------------
    def n_parameters(self) -> Dict[str, int]:
        macro = sum(p.numel() for p in self.high_level_policy.parameters())
        micro = [sum(p.numel() for p in m.parameters()) for m in self.micro_policies]
        return {'macro': macro, 'micro_each': micro[0], 'micro_total': sum(micro), 'total': macro + sum(micro)}

    def hparams(self) -> Dict:
        """Values for Table M1 (HBA-RL columns)."""
        return {
            'algorithm': 'PPO (clipped surrogate, GAE)',
            'update_interval_episodes': 10,
            'minibatch_size': 'full batch (all transitions since the last update; ~1000 per update)',
            'epochs_per_update': self.k_epochs,
            'learning_rate': 3e-4, 'lr_schedule': 'cosine annealing to 1e-5',
            'cosine_decay_horizon_updates': self.total_episodes // 10,
            'adam_eps': 1e-5, 'gamma': self.gamma, 'gae_lambda': self.gae_lambda,
            'clip_eps': self.eps_clip, 'value_loss_coef': self.value_loss_coef,
            'value_clipping': 'yes (same clip eps)',
            'entropy_coef_start': self.entropy_coef_max, 'entropy_coef_end': self.entropy_coef_min,
            'entropy_schedule': 'linear over total_episodes/10 updates',
            'max_grad_norm': self.max_grad_norm,
            'kl_threshold': self.target_kl, 'kl_stop_rule': 'stop epochs when approx KL > 1.5 x threshold',
            'reward_normalisation': 'running mean/std (Welford), applied to stored rewards',
            'macro_routing_loss_weight': 2.0,
            'init': 'orthogonal (gain sqrt2; heads 0.01; value 1.0 for micro)',
            'initial_log_sd_micro': 0.0, 'log_sd_clamp': '[-3, 1]',
            'zero_subgoal': self.zero_subgoal,
            'n_parameters': self.n_parameters(),
        }

    # ------------------------------------------------------------------
    def restore_best(self):
        if self.best_high_state:
            self.high_level_policy.load_state_dict(self.best_high_state)
            for i, p in enumerate(self.micro_policies):
                if self.best_micro_states[i]:
                    p.load_state_dict(self.best_micro_states[i])
            print(f"[Agent] Restored best model (reward={self.best_reward:.2f})")
        else:
            print("[Agent] No best model recorded yet.")

    # ------------------------------------------------------------------
    def save_models(self, path: str):
        torch.save({
            'high_level_policy':   self.high_level_policy.state_dict(),
            'micro_policies':      [p.state_dict() for p in self.micro_policies],
            'high_opt':            self.high_opt.state_dict(),
            'micro_opts':          [o.state_dict() for o in self.micro_opts],
            'best_reward':         self.best_reward,
            'best_high_state':     self.best_high_state,
            'best_micro_states':   self.best_micro_states,
            'reward_rms_mean':     self.reward_rms.mean,
            'reward_rms_var':      self.reward_rms.var,
            'entropy_coef':        self.entropy_coef,
            'update_count':        self.update_count,
            'routing_counts':      self.routing_counts.tolist(),
        }, path)

    def load_models(self, path: str):
        ck = torch.load(path, map_location='cpu')
        self.high_level_policy.load_state_dict(ck['high_level_policy'])
        for i, p in enumerate(self.micro_policies):
            p.load_state_dict(ck['micro_policies'][i])
        self.high_opt.load_state_dict(ck['high_opt'])
        for i, o in enumerate(self.micro_opts):
            o.load_state_dict(ck['micro_opts'][i])
        self.best_reward       = ck.get('best_reward', -np.inf)
        self.best_high_state   = ck.get('best_high_state')
        self.best_micro_states = ck.get('best_micro_states', [None] * self.num_nodes)
        self.reward_rms.mean   = ck.get('reward_rms_mean', 0.0)
        self.reward_rms.var    = ck.get('reward_rms_var', 1.0)
        self.entropy_coef      = ck.get('entropy_coef', self.entropy_coef)
        self.update_count      = ck.get('update_count', 0)
        rc = ck.get('routing_counts')
        if rc:
            self.routing_counts = np.array(rc)