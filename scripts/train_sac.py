# scripts/train_sac.py
"""SAC training for spacecraft attitude takeover (V2 environment)."""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2
from algorithms.sac_agent import SACAgent, ReplayBuffer


def train(args):
    env = SpacecraftTakeoverEnvV2(max_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=act_dim, hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        gamma=args.gamma, tau=args.tau,
        target_entropy_coef=args.entropy_coef,
    )
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = open(os.path.join(args.log_dir, "training_log.csv"), "w")
    log_file.write("episode,total_reward,episode_length,success,alpha,actor_loss,critic_loss\n")

    best_reward = -np.inf
    print(f"SAC training: {args.episodes} episodes, max {args.max_steps} steps")
    print(f"Device: {agent.device}  |  Target entropy: {agent.target_entropy:.2f}\n")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        ep_r, step = 0.0, 0
        actor_losses, critic_losses = [], []

        for step in range(args.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            ep_r += reward

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                train_info = agent.update(batch)
                critic_losses.append(train_info["critic_loss"])
                actor_losses.append(train_info["actor_loss"])

            if terminated or truncated:
                break

        success = info.get("target_fuel", 1.0) <= 0.0
        avg_critic = float(np.mean(critic_losses)) if critic_losses else 0.0
        avg_actor  = float(np.mean(actor_losses)) if actor_losses else 0.0
        log_file.write(f"{episode},{ep_r:.4f},{step+1},{int(success)},{agent.alpha:.4f},{avg_actor:.4f},{avg_critic:.4f}\n")
        log_file.flush()

        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1:5d}/{args.episodes}  |  "
                  f"r={ep_r:8.1f}  |  steps={step+1:3d}  |  "
                  f"success={success}  |  best={best_reward:8.1f}  |  alpha={agent.alpha:.3f}")

        if ep_r > best_reward:
            best_reward = ep_r
            agent.save(os.path.join(args.checkpoint_dir, "best.pt"))

        if (episode + 1) % args.save_every == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"episode_{episode+1}.pt"))

    agent.save(os.path.join(args.checkpoint_dir, "final.pt"))
    log_file.close()
    env.close()
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC training for spacecraft takeover V2")
    parser.add_argument("--episodes",       type=int,   default=3000)
    parser.add_argument("--max-steps",      type=int,   default=600)
    parser.add_argument("--hidden-dim",     type=int,   default=256)
    parser.add_argument("--actor-lr",       type=float, default=3e-4)
    parser.add_argument("--critic-lr",      type=float, default=3e-4)
    parser.add_argument("--alpha-lr",       type=float, default=3e-4)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--tau",            type=float, default=0.005)
    parser.add_argument("--entropy-coef",   type=float, default=1.0)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--buffer-size",    type=int,   default=100000)
    parser.add_argument("--save-every",     type=int,   default=500)
    parser.add_argument("--checkpoint-dir", type=str,   default="outputs/checkpoints")
    parser.add_argument("--log-dir",        type=str,   default="outputs/logs")
    args = parser.parse_args()
    train(args)
