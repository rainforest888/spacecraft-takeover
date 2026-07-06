# scripts/quick_diag.py
"""Quick diagnosis: 20 eps per max_steps, immediate output."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent

agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256, fixed_alpha=0.1)
agent.load("outputs/checkpoints/best_sac_v5.pt")
agent.actor.eval()

print(f"{'max_steps':<10} {'succ%':<8} {'sw%':<8} {'avg_fuel_left%':<16} {'avg_step':<10}")
print("-" * 55)

for ms in [600, 500, 450, 400, 350, 300]:
    env = SpacecraftTakeoverEnvV5(max_steps=ms)
    succ, sw, fl, steps = 0, 0, [], []
    for ep in range(20):
        obs, info = env.reset()
        for s in range(ms):
            obs, r, t, tr, info = env.step(agent.select_action(obs, deterministic=True))
            if t or tr:
                break
        if info.get("target_fuel", 1) <= 0:
            succ += 1
        if info.get("phase_switched"):
            sw += 1
        fl.append(info.get("target_fuel", 0))
        steps.append(s + 1)
    env.close()
    af = np.mean([f for f in fl if f > 0]) * 100 if succ < 20 else 0
    as_ = np.mean(steps)
    print(f"{ms:<10} {succ/20*100:<8.0f} {sw/20*100:<8.0f} {af:<16.1f} {as_:<10.0f}")
    sys.stdout.flush()

print("\nDone.")
