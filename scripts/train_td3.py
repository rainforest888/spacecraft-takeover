# scripts/train_td3.py
"""TD3 training for V5 spacecraft takeover — claw mechanism + efficiency reward."""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.td3_agent import TD3Agent, ReplayBuffer


def train(args):
    env = SpacecraftTakeoverEnvV5(max_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TD3Agent(
        obs_dim=obs_dim, action_dim=act_dim, hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        gamma=args.gamma, tau=args.tau,
        policy_noise=args.policy_noise, noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
    )
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "training_log_v5.csv")
    log_file = open(log_path, "w")
    log_file.write("episode,total_reward,episode_length,success,actor_loss,critic_loss,"
                   "dry_mass,fuel_mass_init,self_fuel_end,phase_switched\n")
    log_file.flush()

    best_reward = -np.inf
    success_count = 0
    # Exploration noise: linear decay
    expl_start = args.expl_noise
    expl_end = 0.05

    print(f"TD3 V5 training: {args.episodes} episodes, max {args.max_steps} steps")
    print(f"Obs dim: {obs_dim}  |  Act dim: {act_dim}")
    print(f"Device: {agent.device}  |  Model: claw_body.xml")
    print(f"Exploration noise: {expl_start:.2f} → {expl_end:.2f}\n")

    for episode in range(args.episodes):
        # Linear noise decay
        frac = episode / max(args.episodes - 1, 1)
        noise_std = expl_start + (expl_end - expl_start) * frac

        obs, info = env.reset()
        ep_r, step = 0.0, 0
        actor_losses, critic_losses = [], []
        switched = False

        for step in range(args.max_steps):
            action = agent.select_action(obs, noise_std=noise_std)
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
                if train_info["actor_loss"] is not None:
                    actor_losses.append(train_info["actor_loss"])

            if terminated or truncated:
                break

        success = info.get("target_fuel", 1.0) <= 0.0
        if success:
            success_count += 1

        avg_actor  = float(np.mean(actor_losses)) if actor_losses else 0.0
        avg_critic = float(np.mean(critic_losses)) if critic_losses else 0.0
        log_file.write(f"{episode},{ep_r:.4f},{step+1},{int(success)},{avg_actor:.4f},"
                       f"{avg_critic:.4f},"
                       f"{info.get('dry_mass', 0):.0f},{info.get('fuel_mass', 0):.0f},"
                       f"{info.get('self_fuel', 0):.4f},{int(switched)}\n")
        log_file.flush()

        if (episode + 1) % 10 == 0:
            rate = success_count / (episode + 1) * 100
            print(f"Ep {episode+1:5d}/{args.episodes}  |  "
                  f"r={ep_r:8.1f}  |  steps={step+1:3d}  |  "
                  f"succ={rate:.1f}% [{success_count}]  |  "
                  f"noise={noise_std:.3f}  |  "
                  f"a_loss={avg_actor:.3f}  c_loss={avg_critic:.3f}")

        if ep_r > best_reward:
            best_reward = ep_r
            agent.save(os.path.join(args.checkpoint_dir, "best_v5.pt"))

        if (episode + 1) % args.save_every == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"v5_ep{episode+1}.pt"))

    agent.save(os.path.join(args.checkpoint_dir, "final_v5.pt"))
    log_file.close()
    env.close()
    final_rate = success_count / args.episodes * 100
    print(f"\nTD3 V5 complete. Best reward: {best_reward:.2f}  |  "
          f"Final success rate: {final_rate:.1f}% [{success_count}/{args.episodes}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 training for V5 spacecraft takeover")
    parser.add_argument("--episodes",       type=int,   default=1000)
    parser.add_argument("--max-steps",      type=int,   default=600)
    parser.add_argument("--hidden-dim",     type=int,   default=256)
    parser.add_argument("--actor-lr",       type=float, default=1e-4)
    parser.add_argument("--critic-lr",      type=float, default=3e-4)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--tau",            type=float, default=0.005)
    parser.add_argument("--policy-noise",   type=float, default=0.2)
    parser.add_argument("--noise-clip",     type=float, default=0.5)
    parser.add_argument("--policy-delay",   type=int,   default=2)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--buffer-size",    type=int,   default=100000)
    parser.add_argument("--expl-noise",     type=float, default=0.3)
    parser.add_argument("--save-every",     type=int,   default=200)
    parser.add_argument("--checkpoint-dir", type=str,   default="outputs/checkpoints")
    parser.add_argument("--log-dir",        type=str,   default="outputs/logs")
    args = parser.parse_args()
    train(args)
