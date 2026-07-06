# scripts/sweep_maxsteps_reward.py
"""Quick sweep: train 200eps × multiple (max_steps, W_SELF_BURN) combinations.
Tests whether shortening the time limit + increasing self-burn penalty
forces the agent to learn an efficient mass-inference strategy.

Usage:
  python scripts/sweep_maxsteps_reward.py
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algorithms.sac_agent import SACAgent, ReplayBuffer


def train_one_config(config, args):
    """Import env here to avoid MuJoCo memory issues."""
    # Patch env class with new defaults
    import envs.spacecraft_env_v5 as env_mod

    max_steps = config["max_steps"]
    w_self_burn = config["w_self_burn"]
    label = config["label"]

    # We need to override the env's class-level constants.
    # Save originals
    orig_max_steps_cls = None  # not a class attr, passed to __init__
    orig_W_SELF_BURN = env_mod.SpacecraftTakeoverEnvV5.W_SELF_BURN

    env_mod.SpacecraftTakeoverEnvV5.W_SELF_BURN = w_self_burn

    env = env_mod.SpacecraftTakeoverEnvV5(max_steps=max_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=act_dim, hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        gamma=args.gamma, tau=args.tau,
        fixed_alpha=0.1,
    )
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    log_dir = "outputs/logs/sweep"
    ckpt_dir = "outputs/checkpoints/sweep"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{label}.csv")
    successes = 0

    with open(log_path, "w") as log_file:
        log_file.write("episode,total_reward,steps,success,phase_switched,self_fuel_end\n")

        for ep in range(args.episodes):
            obs, info = env.reset()
            ep_r = 0.0
            switched = False

            for step in range(max_steps):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                buffer.store(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs
                ep_r += reward

                if info.get("phase_switched", False):
                    switched = True

                if len(buffer) >= args.batch_size:
                    agent.update(buffer.sample(args.batch_size))

                if terminated or truncated:
                    break

            success = info.get("target_fuel", 1.0) <= 0.0
            if success:
                successes += 1

            log_file.write(f"{ep},{ep_r:.4f},{step+1},{int(success)},"
                           f"{int(switched)},{info.get('self_fuel', 0):.4f}\n")
            log_file.flush()

            if (ep + 1) % 20 == 0:
                rate = successes / (ep + 1) * 100
                print(f"  [{label}] Ep {ep+1:4d}/{args.episodes}  succ={rate:5.1f}%  "
                      f"r={ep_r:8.1f}  sw={switched}")

    env.close()
    # Restore class attrs
    env_mod.SpacecraftTakeoverEnvV5.W_SELF_BURN = orig_W_SELF_BURN

    final_rate = successes / args.episodes * 100
    ckpt_path = os.path.join(ckpt_dir, f"{label}.pt")
    agent.save(ckpt_path)
    return final_rate, successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int, default=200)
    parser.add_argument("--hidden-dim",  type=int, default=256)
    parser.add_argument("--actor-lr",    type=float, default=3e-4)
    parser.add_argument("--critic-lr",   type=float, default=3e-4)
    parser.add_argument("--alpha-lr",    type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--tau",         type=float, default=0.005)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    args = parser.parse_args()

    # Configurations to sweep
    configs = [
        # (label, max_steps, W_SELF_BURN)
        {"label": "baseline",          "max_steps": 600, "w_self_burn": 2.0},
        {"label": "ms400",             "max_steps": 400, "w_self_burn": 2.0},
        {"label": "ms450",             "max_steps": 450, "w_self_burn": 2.0},
        {"label": "ms350_sb3",         "max_steps": 350, "w_self_burn": 3.0},
        {"label": "ms400_sb4",         "max_steps": 400, "w_self_burn": 4.0},
        {"label": "ms450_sb3",         "max_steps": 450, "w_self_burn": 3.0},
        {"label": "ms350",             "max_steps": 350, "w_self_burn": 2.0},
    ]

    print("=" * 70)
    print("  Sweep: max_steps × W_SELF_BURN")
    print(f"  {len(configs)} configs × {args.episodes} episodes each")
    print("=" * 70)

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n--- [{i+1}/{len(configs)}] {cfg['label']}: "
              f"max_steps={cfg['max_steps']}, W_SELF_BURN={cfg['w_self_burn']} ---")
        rate, succ = train_one_config(cfg, args)
        results.append({**cfg, "rate": rate, "successes": succ})
        print(f"  >>> {cfg['label']}: {rate:.1f}% ({succ}/{args.episodes})")

    print("\n" + "=" * 70)
    print("  Final Results")
    print("=" * 70)
    print(f"{'Config':<20} {'max_steps':<12} {'W_SELF_BURN':<14} {'Success':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<20} {r['max_steps']:<12} {r['w_self_burn']:<14} "
              f"{r['rate']:.1f}% ({r['successes']}/{args.episodes})")

    best = max(results, key=lambda r: r['rate'])
    print(f"\n  Best: {best['label']} = {best['rate']:.1f}%")
    print(f"  → Next: run full 1000-ep training with best config")


if __name__ == "__main__":
    main()
