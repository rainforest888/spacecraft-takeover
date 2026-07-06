# scripts/quick_test_maxsteps.py
"""Quick diagnostic: evaluate current V5 model at different max_steps to see ceiling effect."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent


def eval_at_maxsteps(agent, max_steps, episodes=100):
    successes = 0
    phase_switches = 0
    fuel_left = []

    env = SpacecraftTakeoverEnvV5(max_steps=max_steps)
    for ep in range(episodes):
        obs, info = env.reset()

        for step in range(max_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        success = info.get("target_fuel", 1.0) <= 0.0
        if success:
            successes += 1
        if info.get("phase_switched"):
            phase_switches += 1

        tf = info.get("target_fuel", 0)
        fuel_left.append(tf)

    env.close()

    avg_fuel_left = np.mean([f for f in fuel_left if f > 0]) if successes < episodes else 0
    return successes / episodes * 100, phase_switches / episodes * 100, avg_fuel_left * 100


if __name__ == "__main__":
    agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256, fixed_alpha=0.1)
    agent.load("outputs/checkpoints/best_sac_v5.pt")
    agent.actor.eval()

    print("=" * 70)
    print("  Current V5 Model — Sensitivity to max_steps (100 eps each)")
    print("=" * 70)
    print(f"{'max_steps':<12} {'Success%':<12} {'PhaseSwitch%':<16} {'AvgFuelLeft(Fail)':<20}")
    print("-" * 70)

    for max_steps in [600, 500, 450, 400, 350, 300]:
        succ, ps, afl = eval_at_maxsteps(agent, max_steps, episodes=100)
        print(f"{max_steps:<12} {succ:<12.1f} {ps:<16.1f} {afl:<20.1f}")

    print("-" * 70)
    print("  → If success drops sharply at shorter max_steps,")
    print("    retraining with that limit may force efficient strategy.")
