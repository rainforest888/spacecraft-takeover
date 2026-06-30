"""SAC (Soft Actor-Critic) agent with entropy auto-tuning.

References:
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL", 2018
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class GaussianActor(nn.Module):
    """Stochastic actor: outputs (mu, log_std) for each action dimension."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action with reparameterization trick. Returns (action, log_prob, mean)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(std)
        action = mu + std * eps
        action = torch.tanh(action)
        # log_prob with tanh squashing correction
        log_prob = self._log_prob(mu, log_std, eps) - self._tanh_log_prob_correction(action)
        return action, log_prob, mu

    def _log_prob(self, mu: torch.Tensor, log_std: torch.Tensor,
                  eps: torch.Tensor) -> torch.Tensor:
        var = (log_std.exp()) ** 2
        return -0.5 * (eps ** 2 + 2.0 * log_std + np.log(2.0 * np.pi)).sum(dim=-1, keepdim=True)

    def _tanh_log_prob_correction(self, action: torch.Tensor) -> torch.Tensor:
        return (2.0 * (np.log(2.0) - action - F.softplus(-2.0 * action))).sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def store(self, obs, action, reward, next_obs, done):
        idx = self._ptr % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self._ptr += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self.obs[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]).unsqueeze(-1),
            torch.from_numpy(self.next_obs[indices]),
            torch.from_numpy(self.dones[indices]).unsqueeze(-1),
        )

    def __len__(self):
        return self._size


class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 target_entropy_coef: float = 1.0,
                 device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Networks
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Entropy tuning
        self.target_entropy = -target_entropy_coef * action_dim  # -dim(A) * coef
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self._update_count = 0

    def select_action(self, obs: np.ndarray, noise_std: float = 0.0,
                       deterministic: bool = False) -> np.ndarray:
        """Select action. Set deterministic=True for evaluation."""
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            if deterministic:
                mu, _ = self.actor.forward(obs_t)
                action = torch.tanh(mu).cpu().numpy().flatten()
            else:
                action, _, _ = self.actor.sample(obs_t)
                action = action.cpu().numpy().flatten()
        self.actor.train()
        return action

    def update(self, batch) -> dict:
        obs, actions, rewards, next_obs, dones = [b.to(self.device) for b in batch]

        # --- Update critics ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            q1_next = self.critic1_target(next_obs, next_actions)
            q2_next = self.critic2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 5.0)
        self.critic_optimizer.step()

        # --- Update actor ---
        new_actions, log_probs, _ = self.actor.sample(obs)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update alpha (entropy tuning) ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # --- Soft update targets ---
        self._update_count += 1
        for target, source in [(self.critic1_target, self.critic1),
                                (self.critic2_target, self.critic2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.log_alpha = ckpt["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
