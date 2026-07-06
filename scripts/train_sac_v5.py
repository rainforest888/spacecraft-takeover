# scripts/train_sac_v5.py
"""SAC training for V5 spacecraft takeover — claw mechanism + efficiency reward."""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent, ReplayBuffer


def train(args):
    env = SpacecraftTakeoverEnvV5(
        max_steps=args.max_steps,
        w_fuel_burn=args.w_fuel_burn,
        w_self_burn=args.w_self_burn,
        w_att=args.w_att,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=act_dim, hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        gamma=args.gamma, tau=args.tau,
        fixed_alpha=0.1,
    )
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Unique log name based on config
    tag = f"ms{args.max_steps}_fb{args.w_fuel_burn}_sb{args.w_self_burn}_att{args.w_att}"
    log_path = os.path.join(args.log_dir, f"train_{tag}.csv")
    log_file = open(log_path, "w")
    log_file.write("episode,total_reward,episode_length,success,alpha,actor_loss,critic_loss,"
                   "dry_mass,fuel_mass_init,self_fuel_end,phase_switched\n")
    log_file.flush()

    best_reward = -np.inf
    success_count = 0
    print(f"SAC V5 training: {args.episodes} episodes, max {args.max_steps} steps")
    print(f"Config: W_FUEL={env.W_FUEL_BURN}  W_SELF={env.W_SELF_BURN}  W_ATT={env.W_ATT}")
    print(f"Obs dim: {obs_dim}  |  Act dim: {act_dim}")
    print(f"Device: {agent.device}  |  Alpha: {agent.alpha:.3f} (fixed)")
    print(f"Log: {log_path}\n")

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
                       f"{info.get('dry_mass', 0):.0f},{info.get('fuel_mass', 0):.0f},"
                       f"{info.get('self_fuel', 0):.4f},{int(switched)}\n")
        log_file.flush()

        if (episode + 1) % 10 == 0:
            rate = success_count / (episode + 1) * 100
            print(f"Ep {episode+1:5d}/{args.episodes}  |  "
                  f"r={ep_r:8.1f}  |  steps={step+1:3d}  |  "
                  f"succ={rate:.1f}% [{success_count}]  |  "
                  f"alpha={agent.alpha:.3f}  |  "
                  f"a_loss={avg_actor:.3f}  c_loss={avg_critic:.3f}")

        if ep_r > best_reward:
            best_reward = ep_r
            ckpt_name = f"best_{tag}.pt"
            agent.save(os.path.join(args.checkpoint_dir, ckpt_name))

        if (episode + 1) % args.save_every == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"sac_{tag}_ep{episode+1}.pt"))

    agent.save(os.path.join(args.checkpoint_dir, f"final_{tag}.pt"))
    log_file.close()
    env.close()
    final_rate = success_count / args.episodes * 100
    print(f"\nSAC V5 [{tag}] complete. Best reward: {best_reward:.2f}  |  "
          f"Final success rate: {final_rate:.1f}% [{success_count}/{args.episodes}]")
    return final_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC training for V5 spacecraft takeover")
    parser.add_argument("--episodes",       type=int,   default=1000)
    parser.add_argument("--max-steps",      type=int,   default=600)
    parser.add_argument("--w-fuel-burn",    type=float, default=None)  # default uses env's W_FUEL_BURN=10.0
    parser.add_argument("--w-self-burn",    type=float, default=None)  # default uses env's W_SELF_BURN=2.0
    parser.add_argument("--w-att",          type=float, default=None)  # default uses env's W_ATT=1.0
    parser.add_argument("--hidden-dim",     type=int,   default=256)
    parser.add_argument("--actor-lr",       type=float, default=3e-4)
    parser.add_argument("--critic-lr",      type=float, default=3e-4)
    parser.add_argument("--alpha-lr",       type=float, default=3e-4)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--tau",            type=float, default=0.005)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--buffer-size",    type=int,   default=100000)
    parser.add_argument("--save-every",     type=int,   default=200)
    parser.add_argument("--checkpoint-dir", type=str,   default="outputs/checkpoints")
    parser.add_argument("--log-dir",        type=str,   default="outputs/logs")
    args = parser.parse_args()
    train(args)
