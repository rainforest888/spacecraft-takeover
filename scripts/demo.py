# scripts/demo.py
"""Render and record takeover demonstration video."""
import os
import sys
import argparse
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.spacecraft_env import SpacecraftTakeoverEnv
from algorithms.td3_agent import TD3Agent
from algorithms.lqr_controller import LQRController
from algorithms.switch_manager import SwitchManager


def demo(args):
    env = SpacecraftTakeoverEnv(render_mode=None, max_steps=args.max_steps)
    agent = TD3Agent(obs_dim=11, action_dim=3, hidden_dim=args.hidden_dim)
    agent.load(args.checkpoint)
    agent.actor.eval()

    I_body = np.diag([500.0, 541.7, 541.7])
    lqr = LQRController(I_body, max_torque=env.MAX_TORQUE)

    renderer = mujoco.Renderer(env.model, args.width, args.height)
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, "demo.mp4")

    frames = []
    obs, info = env.reset()
    switch = SwitchManager()

    print(f"Recording demo -- target strategy: {info['target_strategy']}")
    for step in range(args.max_steps):
        switch.update(
            tau_target_mag=obs[7],
            attitude_error=np.linalg.norm(obs[0:3]),
            self_fuel=obs[6],
        )

        if switch.phase == "weakening":
            action = agent.select_action(obs, noise_std=0.0)
        else:
            tau_td3 = agent.select_action(obs, noise_std=0.0) * env.MAX_TORQUE
            tau_lqr = lqr.compute(obs[0:3], obs[3:6])
            alpha = switch.get_blend_alpha()
            tau = alpha * tau_td3 + (1.0 - alpha) * tau_lqr
            action = np.clip(tau / env.MAX_TORQUE, -1.0, 1.0)

        obs, _, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data)
        pixels = renderer.render()
        frames.append(pixels)

        if step % 60 == 0:
            print(f"  Step {step}: phase={switch.phase} "
                  f"self_fuel={obs[6]:.3f} "
                  f"att_err={np.linalg.norm(obs[0:3]):.3f}")

        if terminated or truncated:
            print(f"  Final: step={step} success={info['target_fuel'] <= 0}")
            break

    try:
        import imageio
        imageio.mimsave(video_path, frames, fps=args.fps)
        print(f"Video saved to {video_path}")
    except ImportError:
        frame_dir = os.path.join(args.output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        import imageio
        for i, frame in enumerate(frames):
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), frame)
        print(f"Frames saved to {frame_dir}/")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="outputs/videos")
    args = parser.parse_args()
    demo(args)
