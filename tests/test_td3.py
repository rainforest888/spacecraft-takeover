# tests/test_td3.py
import numpy as np
import torch
from algorithms.td3_agent import Actor, Critic, ReplayBuffer, TD3Agent


def test_actor_output_shape():
    actor = Actor(obs_dim=11, action_dim=3, hidden_dim=64)
    obs = torch.randn(4, 11)
    action = actor(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_critic_output_shape():
    critic = Critic(obs_dim=11, action_dim=3, hidden_dim=64)
    obs = torch.randn(4, 11)
    action = torch.randn(4, 3)
    q = critic(obs, action)
    assert q.shape == (4, 1)


def test_replay_buffer_store_sample():
    buffer = ReplayBuffer(capacity=1000, obs_dim=11, action_dim=3)
    obs = np.random.randn(11).astype(np.float32)
    action = np.random.randn(3).astype(np.float32)
    for _ in range(50):
        buffer.store(obs, action, np.random.randn(), obs, False)
    assert len(buffer) == 50
    batch = buffer.sample(32)
    assert batch[0].shape == (32, 11)   # obs
    assert batch[1].shape == (32, 3)    # actions


def test_td3_agent_select_action():
    agent = TD3Agent(obs_dim=11, action_dim=3, hidden_dim=64)
    obs = np.random.randn(11).astype(np.float32)
    action = agent.select_action(obs, noise_std=0.0)
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    # With noise: actions should vary
    actions = [agent.select_action(obs, noise_std=0.1) for _ in range(30)]
    assert not np.allclose(actions[0], actions[-1])


def test_td3_update_reduces_losses():
    agent = TD3Agent(obs_dim=11, action_dim=3, hidden_dim=64,
                     actor_lr=1e-3, critic_lr=1e-3)
    buffer = ReplayBuffer(capacity=10000, obs_dim=11, action_dim=3)
    for _ in range(512):
        obs = np.random.randn(11).astype(np.float32)
        action = np.random.randn(3).astype(np.float32)
        buffer.store(obs, action, np.random.randn(), obs, False)

    # First few updates
    initial_losses = []
    for _ in range(5):
        batch = buffer.sample(256)
        info = agent.update(batch)
        if info['critic_loss'] is not None:
            initial_losses.append(info['critic_loss'])

    # More updates
    later_losses = []
    for _ in range(50):
        batch = buffer.sample(256)
        info = agent.update(batch)
        if info['critic_loss'] is not None:
            later_losses.append(info['critic_loss'])

    assert len(later_losses) > 0
    assert np.mean(later_losses) < np.mean(initial_losses)
