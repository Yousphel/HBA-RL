"""
Flat PPO Baseline Agent  (revision 1: deterministic evaluation, categorical node head for ABL-1,
hparams()/n_parameters() for Table M1)
========================
Thesis: Business-Aware Hierarchical RL for Multi-Tenant E-commerce Workload Scheduling

Role in thesis:
---------------
This is the ABLATION BASELINE — a single-level (flat) PPO agent that directly
outputs scheduling actions without any hierarchical decomposition.

Comparing HBA-RL vs FlatPPO isolates the contribution of the hierarchical
structure. If HBA-RL outperforms FlatPPO, it proves the two-level architecture
adds value beyond simply using PPO.

Architecture difference:
    HBA-RL:   state → HighLevelPolicy (subgoal g) → LowLevelPolicy(state, g) → action
    FlatPPO:  state → SinglePolicy → action   (NO subgoal, NO hierarchy)

All other training details are IDENTICAL to HBA-RL:
    - Same PPO algorithm (clip, GAE, entropy annealing)
    - Same hyperparameters (lr, k_epochs, eps_clip, target_kl)
    - Same reward normalisation
    - Same best-model tracking
    - Same seeds
This ensures any performance difference is due to architecture alone.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from seed_utils import set_global_seed


# ==================== Reward Normalizer (identical to HRL) ====================

class RunningMeanStd:
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


# ==================== Single Flat Policy Network ====================

class FlatPolicy(nn.Module):
    """
    Single-level policy network.
    Input:  state (50-dim)

    categorical_node=False (default, original ablation):
        3 continuous outputs [workload, node, allocation] squashed by a sigmoid;
        the node index is int(a1 * num_nodes).
    categorical_node=True (ABL-1 / --matched):
        2 continuous outputs [workload, allocation] + a 5-way CATEGORICAL node head,
        as the supervisor's ABL-1 specification requires. The stored "raw" action is the
        3-vector [raw_workload, raw_allocation, node_index].
    """

    def __init__(self, state_dim: int = 50, action_dim: int = 3, hidden_dim: int = 256,
                 categorical_node: bool = False, num_nodes: int = 5):
        super().__init__()
        self.categorical_node = categorical_node
        self.num_nodes = num_nodes
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        gauss_dim = 2 if categorical_node else action_dim
        self.action_mean = nn.Linear(hidden_dim // 2, gauss_dim)
        self.log_std     = nn.Parameter(torch.zeros(gauss_dim))
        self.value_head  = nn.Linear(hidden_dim // 2, 1)
        self.node_head   = nn.Linear(hidden_dim // 2, num_nodes) if categorical_node else None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)
        if self.node_head is not None:
            nn.init.orthogonal_(self.node_head.weight, gain=0.01)

    def forward(self, state):
        f      = self.net(state)
        mean   = self.action_mean(f)
        std    = torch.exp(self.log_std.clamp(-3, 1)).expand_as(mean)
        logits = self.node_head(f) if self.node_head is not None else None
        return mean, std, self.value_head(f), logits

    def sample_action(self, state, deterministic: bool = False):
        """Returns (env_action[3] in [0,1], raw, log_prob, value)."""
        mean, std, value, logits = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        raw_g = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(raw_g).sum(-1, keepdim=True)
        if not self.categorical_node:
            return torch.sigmoid(raw_g), raw_g, log_prob, value
        cat = torch.distributions.Categorical(logits=logits)
        node = logits.argmax(-1) if deterministic else cat.sample()
        log_prob = log_prob + cat.log_prob(node).unsqueeze(-1)
        a = torch.sigmoid(raw_g)
        env_action = torch.stack([a[:, 0], (node.float() + 0.5) / self.num_nodes, a[:, 1]], dim=-1)
        raw = torch.cat([raw_g, node.float().unsqueeze(-1)], dim=-1)
        return env_action, raw, log_prob, value

    def evaluate_action(self, state, raw):
        mean, std, value, logits = self.forward(state)
        if not self.categorical_node:
            dist     = torch.distributions.Normal(mean, std)
            return dist.log_prob(raw).sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True), value
        dist = torch.distributions.Normal(mean, std)
        cat  = torch.distributions.Categorical(logits=logits)
        node = raw[:, 2].long()
        log_prob = dist.log_prob(raw[:, :2]).sum(-1, keepdim=True) + cat.log_prob(node).unsqueeze(-1)
        entropy  = dist.entropy().sum(-1, keepdim=True) + cat.entropy().unsqueeze(-1)
        return log_prob, entropy, value


# ==================== Flat PPO Agent ====================

class FlatPPOAgent:
    """
    Single-level PPO agent — ablation baseline for HBA-RL.

    Identical hyperparameters and training procedure to HBA-RL.
    Only difference: no hierarchical decomposition.
    """

    def __init__(self,
                 state_dim:         int   = 50,
                 lr:                float = 3e-4,
                 gamma:             float = 0.99,
                 gae_lambda:        float = 0.95,
                 eps_clip:          float = 0.1,
                 k_epochs:          int   = 4,
                 entropy_coef:      float = 0.05,
                 entropy_coef_min:  float = 0.001,
                 value_loss_coef:   float = 0.5,
                 max_grad_norm:     float = 0.5,
                 target_kl:        float = 0.01,
                 normalize_rewards: bool  = True,
                 total_episodes:    int   = 2000,
                 seed:              int   = 42,
                 hidden_dim:        int   = 256,     # MATCHED: 416 -> ~283,900 params (HBA-RL ~282,800)
                 smoothed_best:     bool  = False,   # MATCHED: True -> 10-update smoothed checkpoint, as HBA-RL
                 categorical_node:  bool  = False):  # MATCHED (ABL-1): categorical node head

        # Seed torch identically to HBA-RL
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
        self.seed = seed

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

        # Single flat policy
        self.categorical_node = categorical_node
        self.policy = FlatPolicy(state_dim, hidden_dim=hidden_dim, categorical_node=categorical_node)
        self.smoothed_best  = smoothed_best
        self._reward_window = []

        # Optimizer + LR scheduler — identical to HBA-RL
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_episodes // 10, eta_min=1e-5)

        # Best model tracking
        self.best_reward     = -np.inf
        self.best_state      = None

        # Reward normaliser
        self.reward_rms = RunningMeanStd()

        self._reset_memory()

    # ------------------------------------------------------------------
    def _reset_memory(self):
        self.memory = {
            'states':      [],
            'actions_raw': [],
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
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())

    def hparams(self) -> dict:
        """Values for Table M1 (Flat PPO columns)."""
        return {
            'algorithm': 'PPO (clipped surrogate, GAE)', 'update_interval_episodes': 10,
            'minibatch_size': 'full batch (all transitions since the last update; ~1000 per update)',
            'epochs_per_update': self.k_epochs, 'learning_rate': 3e-4,
            'lr_schedule': 'cosine annealing to 1e-5',
            'cosine_decay_horizon_updates': self.total_episodes // 10, 'adam_eps': 1e-5,
            'gamma': self.gamma, 'gae_lambda': self.gae_lambda, 'clip_eps': self.eps_clip,
            'value_loss_coef': self.value_loss_coef, 'value_clipping': 'yes (same clip eps)',
            'entropy_coef_start': self.entropy_coef_max, 'entropy_coef_end': self.entropy_coef_min,
            'entropy_schedule': 'linear over total_episodes/10 updates', 'max_grad_norm': self.max_grad_norm,
            'kl_threshold': self.target_kl, 'kl_stop_rule': 'stop epochs when approx KL > 1.5 x threshold',
            'reward_normalisation': 'running mean/std (Welford)',
            'init': 'orthogonal (gain sqrt2; action head 0.01; value 1.0)', 'initial_log_sd': 0.0,
            'log_sd_clamp': '[-3, 1]', 'hidden_dim': self.policy.net[0].out_features,
            'node_output': 'categorical head (5 logits)' if self.categorical_node
                           else 'continuous: node = int(a1 * 5)',
            'checkpoint_rule': ('mean of the last 10 update rewards' if self.smoothed_best
                                else 'reward of ONE episode (the last of each 10) - noisy'),
            'n_parameters': self.n_parameters(),
        }

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action directly from state — no hierarchy. deterministic=True: mean / argmax (evaluation)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, raw, log_prob, value = self.policy.sample_action(state_t, deterministic=deterministic)
        return (action.squeeze(0).numpy(),
                raw.squeeze(0),
                log_prob.squeeze(),
                value.squeeze())

    # ------------------------------------------------------------------
    def store_transition(self, state, action_raw, log_prob, value, reward, done):
        self.memory['states'].append(state)
        self.memory['actions_raw'].append(action_raw)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

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
    def update_policy(self, episode_reward: float = None):
        if not self.memory['rewards']:
            return

        self.update_count += 1
        self._anneal_entropy()

        rewards = self.memory['rewards']
        if self.normalize_rewards:
            self.reward_rms.update(rewards)
            rewards = [self.reward_rms.normalize(r) for r in rewards]

        states   = torch.FloatTensor(np.array(self.memory['states']))
        acts_raw = torch.stack(self.memory['actions_raw'])
        old_lp   = torch.stack(self.memory['log_probs']).detach()

        returns, advantages = self._compute_gae(
            rewards, self.memory['values'], self.memory['dones'])

        for epoch in range(self.k_epochs):
            new_lp, entropy, new_val = self.policy.evaluate_action(states, acts_raw)
            new_lp  = new_lp.reshape(-1)
            new_val = new_val.reshape(-1)
            entropy = entropy.reshape(-1)

            ratio = torch.exp(new_lp - old_lp)

            # KL early stopping
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Value loss clipping
            old_val     = returns.detach()
            val_clipped = old_val + torch.clamp(new_val - old_val, -self.eps_clip, self.eps_clip)
            value_loss  = torch.max(F.mse_loss(new_val, returns),
                                    F.mse_loss(val_clipped, returns))

            loss = (-torch.min(surr1, surr2).mean()
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef    * entropy.mean())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()

        # Best model tracking
        score = episode_reward
        if self.smoothed_best:
            # MATCHED: same rule as HBA-RL — mean of the last 10 update rewards
            self._reward_window.append(episode_reward if episode_reward is not None else 0.0)
            if len(self._reward_window) > 10:
                self._reward_window.pop(0)
            score = sum(self._reward_window) / len(self._reward_window)
        if score is not None and score > self.best_reward:
            self.best_reward = score
            self.best_state  = {k: v.cpu().clone()
                                for k, v in self.policy.state_dict().items()}

        self._reset_memory()

    # ------------------------------------------------------------------
    def restore_best(self):
        if self.best_state:
            self.policy.load_state_dict(self.best_state)
            print(f"[FlatPPO] Restored best model (reward={self.best_reward:.2f})")
        else:
            print("[FlatPPO] No best model recorded.")

    # ------------------------------------------------------------------
    def save_model(self, path: str):
        torch.save({
            'policy':          self.policy.state_dict(),
            'optimizer':       self.optimizer.state_dict(),
            'best_reward':     self.best_reward,
            'best_state':      self.best_state,
            'reward_rms_mean': self.reward_rms.mean,
            'reward_rms_var':  self.reward_rms.var,
            'entropy_coef':    self.entropy_coef,
            'update_count':    self.update_count,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.best_reward     = ckpt.get('best_reward', -np.inf)
        self.best_state      = ckpt.get('best_state')
        self.reward_rms.mean = ckpt.get('reward_rms_mean', 0.0)
        self.reward_rms.var  = ckpt.get('reward_rms_var',  1.0)
        self.entropy_coef    = ckpt.get('entropy_coef', self.entropy_coef)
        self.update_count    = ckpt.get('update_count', 0)
