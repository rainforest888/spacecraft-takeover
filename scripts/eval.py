# scripts/eval.py
"""Evaluate trained TD3 + LQR takeover pipeline."""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.spacecraft_env import SpacecraftTakeoverEnv
from algorithms.td3_agent import TD3Agent
from algorithms.lqr_controller import LQRController
from algorithms.switch_manager import SwitchManager


def evaluate(args):
    env = SpacecraftTakeoverEnv(render_mode=None, max_steps=args.max_steps)
    agent = TD3Agent(obs_dim=11, action_dim=3, hidden_dim=args.hidden_dim)
    agent.load(args.checkpoint)
    agent.actor.eval()

    I_body = np.diag([500.0, 541.7, 541.7])
    lqr = LQRController(I_body, max_torque=env.MAX_TORQUE)

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        switch = SwitchManager()
        episode_reward = 0.0
        success = False

        for step in range(args.max_steps):
            switch.update(
                tau_target_mag=obs[7],
                attitude_error=np.linalg.norm(obs[0:3]),
                self_fuel=obs[6],
            )

            if switch.phase == "weakening":
                action = agent.select_action(obs, noise_std=0.0)
                tau = action * env.MAX_TORQUE
            else:
                tau_td3 = agent.select_action(obs, noise_std=0.0) * env.MAX_TORQUE
                tau_lqr = lqr.compute(obs[0:3], obs[3:6])
                alpha = switch.get_blend_alpha()
                tau = alpha * tau_td3 + (1.0 - alpha) * tau_lqr
                action = np.clip(tau / env.MAX_TORQUE, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if info["target_fuel"] <= 0:
                success = True

            if terminated or truncated:
                break

        results.append({
            "episode": ep,
            "reward": episode_reward,
            "steps": step + 1,
            "success": success,
            "final_attitude_error": float(np.linalg.norm(obs[0:3])),
            "target_strategy": info["target_strategy"],
        })
        print(f"[Ep {ep}] reward={episode_reward:.1f} success={success} "
              f"steps={step+1} att_err={np.linalg.norm(obs[0:3]):.4f} "
              f"target={info['target_strategy']}")

    successes = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    print(f"\n=== Summary: {args.episodes} episodes ===")
    print(f"Success rate: {successes}/{args.episodes} "
          f"({100 * successes / max(args.episodes, 1):.1f}%)")
    print(f"Average reward: {avg_reward:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()
    evaluate(args)
