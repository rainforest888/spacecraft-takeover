# scripts/sweep_v2.py
"""Sequential sweep: train 200eps each for key (max_steps, W_SELF, W_ATT) configs."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent, ReplayBuffer


def train_one(label, max_steps, w_fuel_burn, w_self_burn, w_att, episodes=200):
    env = SpacecraftTakeoverEnvV5(
        max_steps=max_steps,
        w_fuel_burn=w_fuel_burn,
        w_self_burn=w_self_burn,
        w_att=w_att,
    )
    agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256,
                     actor_lr=3e-4, critic_lr=3e-4,
                     gamma=0.99, tau=0.005, fixed_alpha=0.1)
    buffer = ReplayBuffer(100000, 10, 3)

    log_dir = "outputs/logs/sweep"
    ckpt_dir = "outputs/checkpoints/sweep"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{label}.csv")
    log = open(log_path, "w")
    log.write("episode,total_reward,steps,success,switched,self_fuel_end,dry_mass,fuel_mass\n")

    successes = 0
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  [{label}] ms={max_steps}  FB={w_fuel_burn}  SB={w_self_burn}  ATT={w_att}")
    print(f"  Log: {log_path}")

    for ep in range(episodes):
        obs, info = env.reset()
        switched = False
        total_r = 0.0
        last_step = 0

        for s in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_r += reward
            last_step = s + 1
            if info.get("phase_switched"):
                switched = True
            if len(buffer) >= 256:
                agent.update(buffer.sample(256))
            if terminated or truncated:
                break

        success = int(info.get("target_fuel", 1.0) <= 0.0)
        successes += success
        log.write(f"{ep},{total_r:.4f},{last_step},{success},{int(switched)},"
                  f"{info.get('self_fuel',0):.4f},{info.get('dry_mass',0):.0f},"
                  f"{info.get('fuel_mass',0):.0f}\n")
        # Flush every 10 eps
        if (ep + 1) % 10 == 0:
            log.flush()

        if (ep + 1) % 20 == 0:
            rate = successes / (ep + 1) * 100
            elapsed = time.time() - t0
            print(f"  Ep {ep+1:4d}/{episodes}  succ={rate:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  est_remain={elapsed/(ep+1)*(episodes-ep-1):.0f}s")

    log.close()
    agent.save(os.path.join(ckpt_dir, f"{label}.pt"))
    env.close()

    rate = successes / episodes * 100
    elapsed = time.time() - t0
    print(f"  >>> [{label}] {rate:.1f}% ({successes}/{episodes})  in {elapsed:.0f}s")
    return rate


if __name__ == "__main__":
    configs = [
        # (label,          max_steps, W_FUEL, W_SELF, W_ATT)
        ("ms400_sb4",      400,       10.0,   4.0,    1.0),
        ("ms450_sb3",      450,       10.0,   3.0,    1.0),
        ("ms400_sb3_att3", 400,       10.0,   3.0,    3.0),
        ("ms450_sb3_att3", 450,       10.0,   3.0,    3.0),
        ("ms400_sb4_att3", 400,       10.0,   4.0,    3.0),
        ("baseline_600",   600,       10.0,   2.0,    1.0),
    ]

    print("Sweep V2: 6 configs × 200 episodes")
    print(f"Estimated total: ~{len(configs) * 15} min\n")

    results = []
    for i, (label, ms, fb, sb, att) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Starting {label}...")
        rate = train_one(label, ms, fb, sb, att, episodes=200)
        results.append((label, ms, fb, sb, att, rate))

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"{'Config':<20} {'ms':<6} {'FB':<6} {'SB':<6} {'ATT':<6} {'Success':<10}")
    print("-" * 56)
    best_rate = 0
    best_label = ""
    for label, ms, fb, sb, att, rate in results:
        print(f"{label:<20} {ms:<6} {fb:<6} {sb:<6} {att:<6} {rate:.1f}%")
        if rate > best_rate:
            best_rate = rate
            best_label = label
    print(f"\n  Best: {best_label} = {best_rate:.1f}%")
