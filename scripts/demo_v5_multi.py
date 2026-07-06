# scripts/demo_v5_multi.py
"""Render V5 demo from three different camera angles and two seeds — quick batch."""
import os, sys, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent


def render_episode(env, agent, renderer, cam, brightness=1.8, with_overlay=True, fps=60):
    """Render one episode. Returns list of frames."""
    frames = []
    obs, info = env.reset()
    terminated, truncated = False, False

    for step in range(env.max_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Dynamic camera orbit
        if cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            cam.azimuth += 0.05

        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render()

        if brightness != 1.0:
            frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if with_overlay:
            frame = _osd(frame, info, step, fps)

        frames.append(frame)

        if terminated or truncated:
            break

    # Freeze last frame
    for _ in range(fps * 2):
        frames.append(frames[-1].copy())

    success = info.get("target_fuel", 1.0) <= 0.0
    return frames, success, info


def _osd(frame, info, step, fps):
    """Minimal telemetry overlay."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("consola.ttf", 26)
        font_big = ImageFont.truetype("consola.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
        font_big = ImageFont.load_default()

    m = 22
    y = m
    for line in [
        f"Target Fuel: {info.get('target_fuel', 0)*100:.0f}%",
        f"Self Fuel:   {info.get('self_fuel', 0)*100:.0f}%",
        f"Att Err:     {np.rad2deg(info.get('attitude_error', 0)):.0f} deg",
        f"Phase:       {'SWITCHED' if info.get('phase_switched') else 'weakening'}",
        f"Step:        {step}  ({step/fps:.1f}s)",
    ]:
        draw.text((m, y), line, fill=(0, 255, 0), font=font)
        y += 30

    # Status banner
    if info.get('target_fuel', 1.0) <= 0.0:
        banner = "TAKEOVER SUCCESS"
        color = (0, 255, 0)
    elif info.get('phase_switched'):
        banner = "PHASE SWITCHED"
        color = (255, 255, 0)
    else:
        banner = "WEAKENING PHASE"
        color = (180, 180, 180)

    bw = 320
    bx = (frame.shape[1] - bw) / 2
    by = frame.shape[0] - m - 50
    draw.rectangle([bx - 8, by - 4, bx + bw + 8, by + 40], fill=(0, 0, 0, 180))
    draw.text((bx + 8, by), banner, fill=color, font=font_big)

    return np.array(img)


def make_camera(lookat, distance, elevation, azimuth):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    return cam


def main():
    W, H = 1280, 720
    FPS = 60
    out_dir = "outputs/videos"
    os.makedirs(out_dir, exist_ok=True)

    agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256, fixed_alpha=0.1)
    agent.load("outputs/checkpoints/best_sac_v5.pt")
    agent.actor.eval()

    configs = [
        # (name, seed, camera)
        ("demo_angle1", 42, make_camera([0.5, 0.0, 0.0], 5.0, -15, 135)),
        ("demo_angle2", 88, make_camera([0.5, 0.3, 0.3], 3.5, -30, 90)),
        ("demo_angle3", 123, make_camera([1.0, -0.5, 0.5], 3.0, -5, 160)),
    ]

    renderer = mujoco.Renderer(mujoco.MjModel.from_xml_path(
        os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "claw_body.xml")),
        W, H)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0

    for name, seed, cam in configs:
        print(f"\nRendering {name} (seed={seed})...")
        env = SpacecraftTakeoverEnvV5(max_steps=600)

        t0 = time.time()
        frames, success, info = render_episode(env, agent, renderer, cam,
                                               brightness=1.8, with_overlay=True, fps=FPS)

        elapsed = time.time() - t0
        print(f"  {len(frames)} frames in {elapsed:.1f}s — {'SUCCESS' if success else 'FAIL'}")

        # Save
        path = os.path.join(out_dir, f"{name}.mp4")
        import imageio
        imageio.mimsave(path, frames, fps=FPS, quality=8, macro_block_size=1)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Saved: {path} ({size_mb:.1f} MB, {len(frames)/FPS:.1f}s)")

        env.close()

    renderer.close()
    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
