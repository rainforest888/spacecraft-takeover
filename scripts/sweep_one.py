# scripts/sweep_one.py
"""Train ONE config and write results to a known log file. Explicit flush everywhere.
Usage: python scripts/sweep_one.py <label> <max_steps> <w_self> <w_att> <w_omega> <episodes>
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent, ReplayBuffer

label = sys.argv[1] if len(sys.argv) > 1 else "test"
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
w_self = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
w_att = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
w_omega = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
episodes = int(sys.argv[6]) if len(sys.argv) > 6 else 100
hidden_dim = int(sys.argv[7]) if len(sys.argv) > 7 else 256
buffer_size = int(sys.argv[8]) if len(sys.argv) > 8 else 100000
utd = int(sys.argv[9]) if len(sys.argv) > 9 else 1
alpha = float(sys.argv[10]) if len(sys.argv) > 10 else 0.1

out_path = f"outputs/logs/sweep_{label}.txt"
os.makedirs("outputs/logs", exist_ok=True)

with open(out_path, "w") as out:
    out.write(f"TRAINING: {label}  ms={max_steps}  sb={w_self}  att={w_att}  omega={w_omega}  eps={episodes}  hdim={hidden_dim}  buf={buffer_size}  utd={utd}  alpha={alpha}\n")
    out.flush()

    env = SpacecraftTakeoverEnvV5(max_steps=max_steps, w_self_burn=w_self, w_att=w_att, w_omega=w_omega)
    fixed_alpha_val = None if alpha < 0 else alpha
    agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=hidden_dim, fixed_alpha=fixed_alpha_val)
    buf = ReplayBuffer(buffer_size, 10, 3)
    successes = 0
    best_succ_rate = 0.0
    best_ep = 0
    t0 = time.time()

    for ep in range(episodes):
        obs, info = env.reset()
        total_r = 0.0
        switched = False
        for s in range(max_steps):
            action = agent.select_action(obs)
            nobs, r, t, tr, info = env.step(action)
            buf.store(obs, action, r, nobs, t or tr)
            obs = nobs
            total_r += r
            if info.get("phase_switched"): switched = True
            if len(buf) >= 256:
                for _ in range(utd):
                    agent.update(buf.sample(256))
            if t or tr: break
        success = int(info.get("target_fuel", 1.0) <= 0.0)
        successes += success
        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = successes / (ep + 1) * 100
            if rate > best_succ_rate:
                best_succ_rate = rate
                best_ep = ep + 1
                agent.save(os.path.join("outputs/checkpoints/sweep", f"best_{label}.pt"))
            line = (f"Ep {ep+1:4d}/{episodes}  succ={rate:.1f}%  "
                    f"elapsed={elapsed:.0f}s  eta={elapsed/(ep+1)*(episodes-ep-1):.0f}s")
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)

    env.close()
    rate = successes / episodes * 100
    elapsed = time.time() - t0

    # Save best + final checkpoints
    ckpt_dir = "outputs/checkpoints/sweep"
    os.makedirs(ckpt_dir, exist_ok=True)
    agent.save(os.path.join(ckpt_dir, f"best_{label}.pt"))   # already saved during training at peak
    agent.save(os.path.join(ckpt_dir, f"final_{label}.pt"))   # final model after all training

    final = f">>> [{label}] {rate:.1f}% ({successes}/{episodes}) in {elapsed:.0f}s  best_so_far={best_succ_rate:.1f}% @ ep{best_ep}"
    out.write(final + "\n")
    print(final, flush=True)
