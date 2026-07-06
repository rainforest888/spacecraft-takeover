# scripts/demo_v5.py
"""Render V5 claw mechanism takeover demo — 1080p60 with telemetry overlay."""
import os, sys, argparse, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent


def draw_overlay(frame, info, step, fps):
    """Draw telemetry overlay on frame using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame  # no overlay if PIL unavailable

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to get a font, fall back to default
    try:
        font_lg = ImageFont.truetype("consola.ttf", 28)
        font_sm = ImageFont.truetype("consola.ttf", 20)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    margin = 24
    y = margin
    line_h = 34

    # ── left panel: state ──
    t = step / fps
    lines_left = [
        f"Time: {t:.1f}s  |  Step: {step}",
        f"Target Fuel:   {info.get('target_fuel', 0)*100:.0f}%",
        f"Self Fuel:     {info.get('self_fuel', 0)*100:.0f}%",
        f"Attitude Err:  {np.rad2deg(info.get('attitude_error', 0)):.1f} deg",
        f"Phase Switch:  {'YES' if info.get('phase_switched', False) else 'NO'}",
    ]
    for line in lines_left:
        draw.text((margin, y), line, fill=(0, 255, 0), font=font_sm)
        y += line_h

    # ── right panel: mass info ──
    y = margin
    lines_right = [
        f"Dry Mass:     {info.get('dry_mass', 0):.0f} kg",
        f"Fuel Mass:    {info.get('fuel_mass', 0):.1f} kg",
        f"Est Mass:     {info.get('est_mass', 0):.0f} kg",
        f"Mass Err:     {info.get('mass_error', 0):.0f} kg",
        f"Target Burn:  {info.get('target_burn_kg', 0)*1000:.1f} g/step",
    ]
    for line in lines_right:
        tw = draw.textlength(line, font=font_sm) if hasattr(draw, 'textlength') else 300
        draw.text((frame.shape[1] - tw - margin, y), line, fill=(255, 200, 50), font=font_sm)
        y += line_h

    # ── bottom center: status banner ──
    success = info.get('target_fuel', 1.0) <= 0.0
    self_dead = info.get('self_fuel', 1.0) <= 0.0
    if success:
        banner = ">>> TAKEOVER SUCCESS — Target Fuel Depleted <<<"
        color = (0, 255, 0)
    elif self_dead:
        banner = ">>> FAILED — Self Fuel Exhausted <<<"
        color = (255, 50, 50)
    elif info.get('phase_switched'):
        banner = ">>> Phase Switched — Dry Mass Detected — Precision Mode <<<"
        color = (255, 255, 0)
    else:
        banner = ">>> Weakening Phase — Burning Target Fuel <<<"
        color = (200, 200, 200)

    bw = draw.textlength(banner, font=font_lg) if hasattr(draw, 'textlength') else 600
    bx = (frame.shape[1] - bw) / 2
    by = frame.shape[0] - margin - 40
    # Draw banner background
    draw.rectangle([bx - 12, by - 4, bx + bw + 12, by + 38], fill=(0, 0, 0, 180))
    draw.text((bx, by), banner, fill=color, font=font_lg)

    # ── top center: strategy ──
    strat = f"Strategy: {info.get('target_strategy', 'LQR')}"
    sw = draw.textlength(strat, font=font_sm) if hasattr(draw, 'textlength') else 200
    sx = (frame.shape[1] - sw) / 2
    draw.text((sx, margin), strat, fill=(150, 150, 255), font=font_sm)

    return np.array(img)


def demo(args):
    print("=" * 70)
    print("  Spacecraft Takeover V5 — Claw Mechanism Demo")
    print("=" * 70)

    env = SpacecraftTakeoverEnvV5(max_steps=args.max_steps)
    agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=args.hidden_dim, fixed_alpha=0.1)
    agent.load(args.checkpoint)
    agent.actor.eval()
    print(f"  Model: {args.checkpoint}")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"  Max steps: {args.max_steps}")
    print()

    # Create renderer once (not inside env.render() each frame)
    renderer = mujoco.Renderer(env.model, args.width, args.height)
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, "demo_v5.mp4")

    frames = []
    obs, info_init = env.reset(seed=args.seed)
    print(f"  Scene:")
    print(f"    Service Sat: 100 kg (blue box + 4-arm claw cage)")
    print(f"    Target Sat:  {info_init['dry_mass']:.0f} kg dry + {info_init['fuel_mass']:.0f} kg fuel")
    print(f"    Target Strategy: {info_init['target_strategy']}")
    print(f"    Self Fuel Budget: 1.0  ({args.max_steps} steps max)")
    print()

    t0 = time.time()
    t_since_log = 0
    terminated, truncated = False, False
    info = info_init

    # Set up camera: start with a good angle, slowly orbit
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.5, 0.0, 0.0]     # midpoint of claw + target
    cam.distance = 5.0                    # back enough to see everything
    cam.elevation = -15.0                 # slightly above
    cam.azimuth = 135.0                   # angled view
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Improve lighting for space scene visibility
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0

    for step in range(args.max_steps):
        # Agent action (deterministic)
        action = agent.select_action(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Slowly orbit camera for dynamic view
        cam.azimuth = 135.0 + step * 0.05  # ~30 deg/min at 60fps

        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render()

        # Brighten the scene for better visibility in space
        frame = np.clip(frame.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)

        # Apply telemetry overlay
        frame_with_osd = draw_overlay(frame, info, step, args.fps)
        frames.append(frame_with_osd)

        # Progress reporting
        t_since_log += 1
        if t_since_log >= 180 or terminated or truncated:  # every 3s of video
            elapsed = time.time() - t0
            pct = step / args.max_steps * 100
            print(f"  [{step:4d}/{args.max_steps}] {pct:5.1f}%  "
                  f"target_fuel={info.get('target_fuel', 0)*100:5.1f}%  "
                  f"self_fuel={info.get('self_fuel', 0):.3f}  "
                  f"att_err={np.rad2deg(info.get('attitude_error', 0)):5.1f}deg  "
                  f"reward={reward:+.3f}  "
                  f"elapsed={elapsed:.1f}s")
            t_since_log = 0

        if terminated or truncated:
            break

    # Finish
    success = info.get("target_fuel", 1.0) <= 0.0
    result = "SUCCESS" if success else \
             "FAIL (self fuel)" if info.get("self_fuel", 0.0) <= 0 else \
             "TIMEOUT"
    elapsed = time.time() - t0

    print(f"\n  >>> Result: {result}")
    print(f"  >>> Steps: {step+1}  |  Rendering time: {elapsed:.1f}s")
    print(f"  >>> Final target_fuel: {info.get('target_fuel', 0)*100:.1f}%")
    print(f"  >>> Phase switched: {info.get('phase_switched', False)}")

    # Freeze last frame for 2 seconds
    if frames:
        for _ in range(args.fps * 2):
            frames.append(frames[-1].copy())

    renderer.close()
    env.close()

    # ── Save video ──
    print(f"\n  Encoding {len(frames)} frames to MP4...")
    try:
        import imageio
        writer = imageio.get_writer(video_path, fps=args.fps, quality=8, codec='libx264')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"\n  Video saved -> {video_path}")
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        duration = len(frames) / args.fps
        print(f"  Size: {size_mb:.1f} MB  |  Duration: {duration:.1f}s  |  FPS: {args.fps}")
    except Exception as e:
        print(f"\n  MP4 encoding failed ({e}), saving frames instead...")
        frame_dir = os.path.join(args.output_dir, "demo_v5_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            import imageio
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:05d}.png"), frame)
        print(f"  {len(frames)} frames saved -> {frame_dir}/")

    # ── Also save a separate clean render (no overlay) for reference ──
    print("\n  Rendering clean version (no overlay)...")
    _render_clean(env, agent, args, info_init['dry_mass'], info_init['fuel_mass'])
    print("\n  Done.")


def _render_clean(env, agent, args, dry_mass, fuel_mass):
    """Render a clean version without telemetry overlay at higher quality."""
    renderer = mujoco.Renderer(env.model, args.width, args.height)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.5, 0.0, 0.0]
    cam.distance = 5.0
    cam.elevation = -15.0
    cam.azimuth = 135.0
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    frames = []
    obs, info = env.reset(seed=args.seed)

    for step in range(args.max_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        cam.azimuth = 135.0 + step * 0.05
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render())

        if terminated or truncated:
            break

    # Freeze
    for _ in range(args.fps * 2):
        frames.append(frames[-1].copy())

    renderer.close()
    env.close()

    clean_path = os.path.join(args.output_dir, "demo_v5_clean.mp4")
    try:
        import imageio
        imageio.mimsave(clean_path, frames, fps=args.fps, quality=8)
        print(f"  Clean video -> {clean_path}")
    except Exception:
        clean_dir = os.path.join(args.output_dir, "demo_v5_clean_frames")
        os.makedirs(clean_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            import imageio
            imageio.imwrite(os.path.join(clean_dir, f"frame_{i:05d}.png"), frame)
        print(f"  Clean frames -> {clean_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Claw Mechanism Demo")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_sac_v5.pt")
    parser.add_argument("--max-steps",  type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--width",      type=int, default=1920)
    parser.add_argument("--height",     type=int, default=1080)
    parser.add_argument("--fps",        type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="outputs/videos")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    demo(args)
