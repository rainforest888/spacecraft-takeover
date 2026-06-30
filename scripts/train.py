# scripts/train.py
"""Offline TD3 training for spacecraft attitude takeover."""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.spacecraft_env import SpacecraftTakeoverEnv
from algorithms.td3_agent import TD3Agent, ReplayBuffer


def train(args):
    env = SpacecraftTakeoverEnv(render_mode=None, max_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TD3Agent(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        gamma=args.gamma, tau=args.tau,
        policy_noise=args.policy_noise, noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
    )
    buffer = ReplayBuffer(capacity=args.buffer_size, obs_dim=obs_dim, action_dim=action_dim)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_file = open(os.path.join(args.log_dir, "training_log.csv"), "w")
    log_file.write("episode,total_reward,episode_length,success,avg_q\n")

    best_reward = -np.inf
    print(f"Training for {args.episodes} episodes (max {args.max_steps} steps each)")
    print(f"Checkpoints: {args.checkpoint_dir}  |  Logs: {args.log_dir}\n")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_q_values = []

        for step in range(args.max_steps):
            action = agent.select_action(obs, noise_std=args.exploration_noise)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)

            obs = next_obs
            episode_reward += reward

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                train_info = agent.update(batch)
                if train_info["critic_loss"] is not None:
                    episode_q_values.append(train_info["critic_loss"])

            if terminated or truncated:
                break

        avg_q = float(np.mean(episode_q_values)) if episode_q_values else 0.0
        success = info.get("target_fuel", 1.0) <= 0.0
        log_file.write(f"{episode},{episode_reward:.4f},{step + 1},{int(success)},{avg_q:.6f}\n")

        if (episode + 1) % 50 == 0:
            print(f"Ep {episode + 1:5d}/{args.episodes}  |  "
                  f"reward: {episode_reward:8.2f}  |  steps: {step + 1:3d}  |  "
                  f"success: {success}  |  best: {best_reward:8.2f}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.checkpoint_dir, "best.pt"))

        if (episode + 1) % args.save_every == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"episode_{episode + 1}.pt"))

    agent.save(os.path.join(args.checkpoint_dir, "final.pt"))
    log_file.close()
    env.close()
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 training for spacecraft takeover")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--log-dir", type=str, default="outputs/logs")
    args = parser.parse_args()
    train(args)
