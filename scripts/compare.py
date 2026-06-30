# scripts/compare.py
"""Baseline comparison: PID-only vs LQR-only vs TD3+LQR."""
import os, sys, argparse
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env import SpacecraftTakeoverEnv
from algorithms.td3_agent import TD3Agent
from algorithms.lqr_controller import LQRController
from algorithms.switch_manager import SwitchManager


def run_pid_only(env, max_steps, pid_cfg):
    from algorithms.target_controllers import PIDController
    pid = PIDController(**pid_cfg)
    obs, _ = env.reset()
    for step in range(max_steps):
        err = obs[0:3]
        omega = obs[3:6]
        tau = pid.compute(err, omega, dt=1/60)
        action = np.clip(tau / env.MAX_TORQUE, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"success": info["target_fuel"] <= 0,
                    "reward": reward, "steps": step + 1,
                    "att_err": float(np.linalg.norm(obs[0:3]))}
    return {"success": False, "reward": 0, "steps": max_steps,
            "att_err": float(np.linalg.norm(obs[0:3]))}


def run_lqr_only(env, max_steps, lqr):
    obs, _ = env.reset()
    for step in range(max_steps):
        tau = lqr.compute(obs[0:3], obs[3:6])
        action = np.clip(tau / env.MAX_TORQUE, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"success": info["target_fuel"] <= 0,
                    "reward": reward, "steps": step + 1,
                    "att_err": float(np.linalg.norm(obs[0:3]))}
    return {"success": False, "reward": 0, "steps": max_steps,
            "att_err": float(np.linalg.norm(obs[0:3]))}


def run_td3_lqr(env, max_steps, agent, lqr):
    obs, _ = env.reset()
    switch = SwitchManager()
    for step in range(max_steps):
        switch.update(tau_target_mag=obs[7], attitude_error=np.linalg.norm(obs[0:3]),
                      self_fuel=obs[6])
        if switch.phase == "weakening":
            action = agent.select_action(obs, noise_std=0.0)
        else:
            tau_td3 = agent.select_action(obs, noise_std=0.0) * env.MAX_TORQUE
            tau_lqr = lqr.compute(obs[0:3], obs[3:6])
            alpha = switch.get_blend_alpha()
            tau = alpha * tau_td3 + (1.0 - alpha) * tau_lqr
            action = np.clip(tau / env.MAX_TORQUE, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"success": info["target_fuel"] <= 0,
                    "reward": reward, "steps": step + 1,
                    "att_err": float(np.linalg.norm(obs[0:3]))}
    return {"success": False, "reward": 0, "steps": max_steps,
            "att_err": float(np.linalg.norm(obs[0:3]))}


def compare(args):
    env = SpacecraftTakeoverEnv(render_mode=None, max_steps=args.max_steps)
    I_body = np.diag([500.0, 541.7, 541.7])
    lqr = LQRController(I_body, max_torque=env.MAX_TORQUE)

    agent = None
    if args.checkpoint:
        agent = TD3Agent(obs_dim=11, action_dim=3, hidden_dim=args.hidden_dim)
        agent.load(args.checkpoint)
        agent.actor.eval()

    results = defaultdict(list)
    pid_cfg = {"kp": 5.0, "ki": 0.5, "kd": 2.5, "max_torque": env.MAX_TORQUE}

    for ep in range(args.episodes):
        r_pid = run_pid_only(env, args.max_steps, pid_cfg)
        env.reset()
        r_lqr = run_lqr_only(env, args.max_steps, lqr)
        env.reset()

        results["PID"].append(r_pid)
        results["LQR"].append(r_lqr)

        if agent is not None:
            env.reset()
            r_td3lqr = run_td3_lqr(env, args.max_steps, agent, lqr)
            results["TD3+LQR"].append(r_td3lqr)

    print(f"\n{'Method':<12} {'Success%':>10} {'AvgSteps':>10} {'AvgAttErr':>10}")
    print("-" * 45)
    for method, res in results.items():
        succ = 100 * sum(r["success"] for r in res) / len(res)
        steps = np.mean([r["steps"] for r in res])
        att = np.mean([r["att_err"] for r in res])
        print(f"{method:<12} {succ:>9.1f}% {steps:>9.1f} {att:>9.4f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()
    compare(args)
