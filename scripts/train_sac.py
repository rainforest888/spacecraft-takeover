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
        fixed_alpha=0.2,
    )
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = open(os.path.join(args.log_dir, "training_log.csv"), "w")
    log_file.write("episode,total_reward,episode_length,success,alpha,actor_loss,critic_loss,"
                   "dry_mass,est_mass,fuel_mass_init,phase_switched\n")
    log_file.flush()

    best_reward = -np.inf
    success_count = 0
    print(f"SAC training v2: {args.episodes} episodes, max {args.max_steps} steps")
    print(f"Obs dim: {obs_dim}  |  Act dim: {act_dim}")
    print(f"Device: {agent.device}  |  Alpha: {agent.alpha:.3f} (fixed)")
    print(f"log_std_min: -5.0\n")

    for episode in range(args.episodes):
        obs, info = env.reset()
        ep_r, step = 0.0, 0
        actor_losses, critic_losses = [], []
        switched = False

        for step in range(args.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            ep_r += reward

            if info.get("phase_switched", False):
                switched = True

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                train_info = agent.update(batch)
                critic_losses.append(train_info["critic_loss"])
                actor_losses.append(train_info["actor_loss"])

            if terminated or truncated:
                break

        success = info.get("target_fuel", 1.0) <= 0.0
        if success:
            success_count += 1

        avg_critic = float(np.mean(critic_losses)) if critic_losses else 0.0
        avg_actor  = float(np.mean(actor_losses))  if actor_losses  else 0.0
        log_file.write(f"{episode},{ep_r:.4f},{step+1},{int(success)},{agent.alpha:.4f},"
                       f"{avg_actor:.4f},{avg_critic:.4f},"
                       f"{info.get('dry_mass', 0):.0f},{info.get('est_mass', 0):.0f},"
                       f"{info.get('fuel_mass', 0):.0f},{int(switched)}\n")
        log_file.flush()

        if (episode + 1) % 10 == 0:
            rate = success_count / (episode + 1) * 100
            print(f"Ep {episode+1:5d}/{args.episodes}  |  "
                  f"r={ep_r:8.1f}  |  steps={step+1:3d}  |  "
                  f"succ={rate:.0f}% [{success_count}]  |  "
                  f"alpha={agent.alpha:.3f}  |  "
                  f"a_loss={avg_actor:.1f}  c_loss={avg_critic:.1f}")

        if ep_r > best_reward:
            best_reward = ep_r
            agent.save(os.path.join(args.checkpoint_dir, "best.pt"))

        if (episode + 1) % args.save_every == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"episode_{episode+1}.pt"))

    agent.save(os.path.join(args.checkpoint_dir, "final.pt"))
    log_file.close()
    env.close()
    final_rate = success_count / args.episodes * 100
    print(f"\nTraining complete. Best reward: {best_reward:.2f}  |  "
          f"Final success rate: {final_rate:.1f}% [{success_count}/{args.episodes}]")


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
    parser.add_argument("--entropy-coef",   type=float, default=0.3)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--buffer-size",    type=int,   default=100000)
    parser.add_argument("--save-every",     type=int,   default=500)
    parser.add_argument("--checkpoint-dir", type=str,   default="outputs/checkpoints")
    parser.add_argument("--log-dir",        type=str,   default="outputs/logs")
    args = parser.parse_args()
    train(args)
