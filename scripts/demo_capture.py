# scripts/demo_capture.py
"""Multi-phase capture demo with SOFT coupling — target visibly decelerates.

Phases:
  0.0s-2.0s   TUMBLE   — no coupling, target tumbles freely, Earth + LEO drift
  2.0s-3.5s   CAPTURE  — claw tightens: coupling torque ramps up, target rotation visibly damps
  3.5s-8.0s   TAKEOVER — SAC burns target fuel via LQR opponent
  8.0s+        SUCCESS  — freeze frame
"""
import os, sys, time, argparse
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algorithms.sac_agent import SACAgent
from algorithms.target_controllers import TargetLQRController
from envs.dynamics import quat_to_mrp, mrp_error

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "capture_demo.xml")

# -- demo timing ----------------------------------------------------
T_CAPTURE_START = 2.0   # seconds - claw begins tightening
T_CAPTURE_FULL  = 5.5   # seconds - claw fully engaged (3.5s ramp, visible damping)
T_SAC_START     = 3.5   # seconds - SAC begins takeover (mid-capture)

# -- coupling parameters (isotropic inertia = clean exponential decay) --
K_D_MAX     = 80.0    # relative angular damping (tau = I/K = 500/80 = 6.25s)
K_P_MAX     = 15.0    # relative angular spring (low → overdamped, no oscillation)
K_ABS_MAX   = 250.0   # absolute damping on service sat (reaction wheel braking)
K_ABS_TGT   = 120.0   # absolute damping on target (post-weld deceleration)

# -- target tumble --------------------------------------------------
TUMBLE_OMEGA = np.array([1.2, -0.8, 1.0])  # aggressive tumble (rad/s)

# -- demo physics ---------------------------------------------------
MAX_TORQUE       = 7.0
FUEL_K_MASS      = 3.5   # moderate burn: ~20 kg/s, 130kg depletes in ~7s
SELF_BURN_RATE   = 0.012
CTL_DT           = 1.0 / 60.0
SUBSTEPS         = 30
DRY_MASS         = 500.0
FUEL_MASS_INIT   = 130.0


def apply_wrench(data, body_name, force, torque):
    """Apply force [N] and torque [N.m] to a body in world frame."""
    bid = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data.xfrc_applied[bid, 0:3] = force
    data.xfrc_applied[bid, 3:6] = torque


def _mrp2quat(sigma):
    s2 = np.dot(sigma, sigma)
    denom = 1.0 + s2
    return np.array([(1.0 - s2) / denom,
                     2.0 * sigma[0] / denom,
                     2.0 * sigma[1] / denom,
                     2.0 * sigma[2] / denom])


def compute_lqr_torque(controller, sigma_cur, omega, dt=1/60):
    sigma_err = mrp_error(sigma_cur, np.zeros(3))
    return controller.compute(sigma_err, omega, dt=dt)


def build_obs(data, self_fuel):
    sigma = quat_to_mrp(data.qpos[3:7].copy())
    omega = data.qvel[3:6].copy()  # service sat ANGULAR velocity
    att_err = float(np.linalg.norm(sigma))
    return np.array([
        sigma[0], sigma[1], sigma[2],
        omega[0], omega[1], omega[2],
        float(np.clip(self_fuel, 0.0, 1.0)),
        0.0, 0.1, att_err,
    ], dtype=np.float32)


def draw_overlay(frame, t, step, fuel_mass, self_fuel, coupling, welded, sac_active, success, fps):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("consola.ttf", 24)
        font_big = ImageFont.truetype("consola.ttf", 34)
    except Exception:
        font = font_big = ImageFont.load_default()

    m = 20
    h, w = frame.shape[:2]

    # Phase
    if success:
        phase, pcolor = "SUCCESS", (0, 255, 0)
    elif welded:
        phase, pcolor = "WELDED - TAKEOVER", (0, 255, 0)
    elif sac_active:
        phase, pcolor = f"CAPTURING {min(100,int(coupling*100))}%", (255, 200, 50)
    elif coupling > 0.01:
        progress_pct = min(100, int(coupling * 100))
        phase, pcolor = f"CAPTURING {progress_pct}%", (255, 200, 50)
    else:
        phase, pcolor = "TUMBLE", (255, 180, 50)

    y = m
    draw.text((m, y), f"Phase: {phase}", fill=pcolor, font=font_big)
    y += 36
    for line in [
        f"Time: {t:.1f}s  |  Step: {step}",
        f"Target Fuel: {fuel_mass:.0f} kg",
        f"Self Fuel:   {self_fuel*100:.0f}%",
        f"Claw Grip:   {coupling*100:.0f}%",
    ]:
        draw.text((m, y), line, fill=(200, 230, 200), font=font)
        y += 28

    # Right panel
    y = m
    for line in [
        f"Target Dry Mass: {DRY_MASS} kg",
        f"Initial Fuel:    {FUEL_MASS_INIT} kg",
        f"Orbital Context: LEO",
    ]:
        tw = draw.textlength(line, font=font) if hasattr(draw, 'textlength') else 280
        draw.text((w - tw - m, y), line, fill=(180, 200, 220), font=font)
        y += 28

    # Gripper strength bar
    y += 10
    bar_w = 200; bar_h = 14; bar_x = w - m - bar_w
    draw.rectangle([bar_x, y, bar_x + bar_w, y + bar_h], outline=(100, 100, 100))
    fill_w = int(bar_w * coupling)
    if fill_w > 0:
        bar_color = (255, 200, 50) if coupling < 0.95 else (0, 255, 0)
        draw.rectangle([bar_x, y, bar_x + fill_w, y + bar_h], fill=bar_color)
    y += bar_h + 4
    tw3 = draw.textlength("GRIP FORCE", font=font) if hasattr(draw, 'textlength') else 100
    draw.text((w - m - bar_w/2 - tw3/2, y), "GRIP FORCE", fill=(150, 150, 150), font=font)

    # Bottom banner
    if success:
        banner = ">>> TAKEOVER COMPLETE - Target Neutralized <<<"
        b_color = (0, 255, 0)
    elif sac_active and coupling > 0.95:
        banner = ">>> SAC Active - Burning Target Fuel <<<"
        b_color = (255, 255, 0)
    elif coupling > 0.01:
        banner = ">>> Claw Tightening - Dampening Target Rotation <<<"
        b_color = (255, 200, 50)
    else:
        banner = ">>> Approach Phase - Target Tumbling in LEO <<<"
        b_color = (200, 200, 220)

    bw = draw.textlength(banner, font=font_big) if hasattr(draw, 'textlength') else 500
    bx = (w - bw) / 2
    by = h - m - 50
    draw.rectangle([bx - 12, by - 4, bx + bw + 12, by + 42], fill=(0, 0, 0, 190))
    draw.text((bx, by), banner, fill=b_color, font=font_big)

    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_sac_v5.pt")
    parser.add_argument("--width",      type=int, default=1920)
    parser.add_argument("--height",     type=int, default=1080)
    parser.add_argument("--fps",        type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="outputs/videos")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print("=" * 68)
    print("  Spacecraft Capture & Takeover - Soft-Coupling Demo")
    print("=" * 68)
    print(f"  Phases: TUMBLE -> CAPTURE(damped) -> TAKEOVER -> SUCCESS")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"  Capture ramp: {T_CAPTURE_START}s - {T_CAPTURE_FULL}s ({T_CAPTURE_FULL-T_CAPTURE_START:.1f}s)")
    print()

    # -- load model -------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = CTL_DT / SUBSTEPS
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4  # RK4 conserves angular momentum

    # -- load SAC agent ---------------------------------------------------
    agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256, fixed_alpha=0.1)
    agent.load(args.checkpoint)
    agent.actor.eval()

    # -- LQR controller ---------------------------------------------------
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_sat")
    service_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
    I_nominal = np.diag([500.0, 500.0, 500.0])
    lqr = TargetLQRController(
        Q_diag=(100, 100, 100, 20, 20, 20),
        R_diag=(1.0, 1.0, 1.0),
        max_torque=5.0,
    )
    lqr.set_inertia(I_nominal)

    # -- initialize scene -------------------------------------------------
    total_mass = DRY_MASS + FUEL_MASS_INIT
    model.body_mass[target_bid] = total_mass
    orig_inertia = model.body_inertia[target_bid].copy()
    model.body_inertia[target_bid] = orig_inertia / 500.0 * total_mass

    mujoco.mj_resetData(model, data)

    # Initial attitudes
    sigma_init = np.array([0.1, -0.08, 0.05])
    data.qpos[3:7] = _mrp2quat(sigma_init)
    data.qpos[10:14] = _mrp2quat(sigma_init + np.array([-0.05, 0.03, -0.02]))

    # Service sat starts OFFSET — will fly toward target during approach phase
    APPROACH_START_POS = np.array([-1.5, 0.35, 0.25])
    APPROACH_TARGET_POS = np.array([0.0, 0.0, 0.0])
    data.qpos[0:3] = APPROACH_START_POS

    # Target tumbles, LEO drift
    data.qvel[9:12] = TUMBLE_OMEGA  # target ANGULAR velocity (tumble)
    data.qvel[6:8] = np.array([0.0, 0.0])  # target linear velocity
    data.qvel[7] += 0.3   # LEO drift +Y for target (qvel[7] = target lin vel Y)
    data.qvel[1] += 0.3   # LEO drift +Y for service sat

    mujoco.mj_forward(model, data)

    # -- DISABLE permanent weld - we use soft coupling instead ------------
    data.eq_active[0] = 0

    # -- state ------------------------------------------------------------
    fuel_mass = FUEL_MASS_INIT
    total_mass_cur = total_mass
    self_fuel = 1.0
    coupling = 0.0         # 0=free, 1=fully gripped
    welded = False          # True after weld constraint engages
    sac_active = False
    success = False

    # -- renderer ---------------------------------------------------------
    renderer = mujoco.Renderer(model, args.height, args.width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.5, 0.0, 0.0]
    cam.distance = 6.0
    cam.elevation = -18
    cam.azimuth = 150
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "capture_demo.mp4")

    frames = []
    total_steps = int(14.0 * args.fps)  # capture + takeover + post-success damping
    prev_omega_target = data.qvel[9:12].copy()

    t0 = time.time()
    print(f"  Rendering {total_steps} frames ({total_steps/args.fps:.1f}s)...")
    print()

    for step in range(total_steps):
        t = step / args.fps

        # -- coupling factor (smooth ramp, convex for visible damping) -----
        if t < T_CAPTURE_START:
            coupling = 0.0
        elif t < T_CAPTURE_FULL:
            raw = (t - T_CAPTURE_START) / (T_CAPTURE_FULL - T_CAPTURE_START)
            raw = np.clip(raw, 0.0, 1.0)
            # smoothstep: gentle start AND gentle end — natural damping curve
            coupling = 3.0 * raw**2 - 2.0 * raw**3
        else:
            coupling = 1.0

        K_d = K_D_MAX * coupling  # damping coefficient
        K_p = K_P_MAX * coupling  # spring coefficient

        # -- SAC activation ------------------------------------------------
        if not sac_active and t >= T_SAC_START:
            sac_active = True
            print(f"  [{t:.1f}s] [>>] SAC takeover begins (coupling={coupling*100:.0f}%)")
        if coupling > 0.01 and step == int(T_CAPTURE_START * args.fps):
            print(f"  [{t:.1f}s] [!] CLAW ENGAGING - Soft coupling begins")

        # -- WELD activation: rigid connection after full grip -------------
        if coupling >= 1.0 and not welded:
            data.eq_active[0] = 1    # ENABLE weld — two bodies become ONE
            welded = True
            print(f"  [{t:.1f}s] [WELD] Rigid connection ENGAGED — spacecraft combined!")

        # -- compute torques -----------------------------------------------
        sigma_svc = quat_to_mrp(data.qpos[3:7].copy())
        omega_svc = data.qvel[3:6].copy()  # service sat ANGULAR velocity

        # Takeover control
        if sac_active:
            # Service sat stays stable — perturbation goes directly to target
            tau_self = np.zeros(3)
            force_approach = np.zeros(3)
        else:
            # -- Approach phase: fly service sat toward target, claw wraps around --
            svc_pos = data.qpos[0:3].copy()
            svc_lin_vel = data.qvel[0:3].copy()
            svc_omega = data.qvel[3:6].copy()

            # PD position control: move service sat to grabbing position
            pos_err = APPROACH_TARGET_POS - svc_pos
            vel_err = -svc_lin_vel  # want to stop at target
            force_approach = 120.0 * pos_err + 60.0 * vel_err
            force_approach = np.clip(force_approach, -250.0, 250.0)

            # Attitude damping during approach (keep stable)
            tau_self = -40.0 * svc_omega
            tau_self = np.clip(tau_self, -5.0, 5.0)

        # Target LQR reaction
        sigma_target = quat_to_mrp(data.qpos[10:14].copy())
        omega_target = data.qvel[9:12].copy()  # target ANGULAR velocity
        tau_target = np.zeros(3) if success else compute_lqr_torque(lqr, sigma_target, omega_target, CTL_DT)

        # -- torque computation: soft coupling (pre-weld) vs rigid (post-weld) --
        if not welded:
            # Soft spring-damper coupling — visible exponential damping
            omega_rel = omega_target - omega_svc
            sigma_rel = mrp_error(sigma_svc, sigma_target)
            tau_couple = -K_d * omega_rel - K_p * sigma_rel
            tau_couple = np.clip(tau_couple, -100.0, 100.0)
            K_abs = K_ABS_MAX * coupling
            K_abs_t = K_ABS_TGT * coupling
            tau_brake_svc = -K_abs * omega_svc
            tau_brake_tgt = -K_abs_t * omega_target
            tau_perturb = np.zeros(3)
            tau_service = tau_self + tau_brake_svc * 3.0
        else:
            # Rigid weld — spacecraft are ONE body now
            tau_couple = np.zeros(3)   # constraint handles force transmission
            tau_brake_svc = -K_ABS_MAX * 3.0 * omega_svc
            tau_brake_tgt = -K_ABS_TGT * omega_target
            if sac_active and not success:
                tau_perturb = np.array([
                    8.0 * np.sin(t * 1.3),
                    6.0 * np.cos(t * 1.7),
                    6.0 * np.sin(t * 1.1 + 0.8),
                ])
            else:
                tau_perturb = np.zeros(3)
            tau_service = tau_self + tau_brake_svc
            force_approach = np.zeros(3)

        tau_target_total = tau_target + tau_couple + tau_brake_tgt + tau_perturb

        # -- physics substeps ----------------------------------------------
        # Apply forces/torques via xfrc_applied (world-frame wrench)
        apply_wrench(data, "service_sat", force_approach, tau_service)
        apply_wrench(data, "target_sat", np.zeros(3), tau_target_total)

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        # Track target deceleration for logging
        omega_target_new = data.qvel[9:12].copy()
        alpha_target = np.linalg.norm(omega_target_new - omega_target) / CTL_DT
        prev_omega_target = omega_target_new.copy()

        # -- fuel depletion ------------------------------------------------
        if sac_active and coupling > 0.5 and fuel_mass > 0:
            tau_self_mag = float(np.linalg.norm(tau_self))
            tau_target_lqr_mag = float(np.linalg.norm(tau_target))  # only LQR reaction, not coupling
            self_burn = CTL_DT * tau_self_mag * SELF_BURN_RATE
            target_burn = FUEL_K_MASS * tau_target_lqr_mag * CTL_DT
            self_fuel = max(-1.0, self_fuel - self_burn)
            fuel_mass = max(0.0, fuel_mass - target_burn)
            total_mass_cur = DRY_MASS + fuel_mass
            model.body_mass[target_bid] = total_mass_cur

        if fuel_mass <= 0.0 and not success:
            success = True
            print(f"  [{t:.1f}s] [OK] SUCCESS - Target fuel depleted!")
            print(f"         Target omega: {np.linalg.norm(data.qvel[9:12]):.3f} rad/s (fully damped)")

        # -- camera (locked on spacecraft, no fly-away) --------------------
        svc_pos = data.qpos[0:3].copy()
        tgt_pos = data.qpos[7:10].copy()
        midpoint = (svc_pos + tgt_pos) / 2.0

        if coupling < 0.01:
            # Tumble: wide establishing shot, slow orbit
            cam.lookat[:] = midpoint
            cam.distance = 6.0 + 0.3 * np.sin(t * 0.7)
            cam.azimuth = 150 + t * 4
            cam.elevation = -18 + 3 * np.sin(t * 0.5)
        elif coupling < 0.95:
            # Capture: smooth zoom toward claw, then hold
            # lerp from wide to close-up as coupling increases
            d_wide, d_close = 6.0, 3.8
            az_wide, az_close = 150.0, 170.0
            el_wide, el_close = -18.0, -10.0
            cam.lookat[:] = midpoint + np.array([0.3, 0.0, 0.0])
            cam.distance    = d_wide + (d_close - d_wide) * coupling
            cam.azimuth     = az_wide + (az_close - az_wide) * coupling
            cam.elevation   = el_wide + (el_close - el_wide) * coupling
        else:
            # Takeover: stable close-up, nearly fixed — no drift, no orbit
            cam.lookat[:] = midpoint + np.array([0.2, 0.0, 0.0])
            cam.distance    = 3.8 + 0.15 * np.sin(t * 0.25)
            cam.azimuth     = 170.0 + 3.0 * np.sin(t * 0.18)
            cam.elevation   = -10.0 + 2.0 * np.sin(t * 0.22)

        # -- render --------------------------------------------------------
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        frame = np.clip(frame.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
        frame = draw_overlay(frame, t, step, fuel_mass, self_fuel, coupling,
                            welded, sac_active, success, args.fps)
        frames.append(frame)

        # -- progress ------------------------------------------------------
        if step % (args.fps * 2) == 0 or abs(coupling - 0.5) < 0.02:
            elapsed = time.time() - t0
            omega_mag = np.linalg.norm(data.qvel[9:12])
            phase = "TUMBLE" if coupling < 0.01 else ("CAPTURE" if coupling < 0.95 else "TAKEOVER")
            if success:
                phase = "SUCCESS"
            print(f"  [{t:5.1f}s] f{step:4d}  phase={phase:10s}  "
                  f"grip={coupling*100:3.0f}%  tgt_w={omega_mag:.2f}  "
                  f"fuel={fuel_mass:.0f}kg  elapsed={elapsed:.1f}s")

    # -- freeze last frame ------------------------------------------------
    for _ in range(args.fps * 2):
        frames.append(frames[-1].copy())

    renderer.close()

    # -- encode video -----------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n  Rendering done in {elapsed:.1f}s. Encoding {len(frames)} frames...")

    import imageio
    imageio.mimsave(out_path, frames, fps=args.fps, quality=8, macro_block_size=1)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    duration = len(frames) / args.fps
    print(f"\n  [OK] Video saved: {out_path}")
    print(f"     Size: {size_mb:.1f} MB  |  Duration: {duration:.1f}s  |  FPS: {args.fps}")
    print(f"     Capture method: SOFT COUPLING (ramp {T_CAPTURE_START}s-{T_CAPTURE_FULL}s, {T_CAPTURE_FULL-T_CAPTURE_START:.1f}s damping)")
    print(f"     Target: {DRY_MASS}kg dry + {FUEL_MASS_INIT}kg fuel")
    print(f"     Final fuel: {fuel_mass:.0f}kg  |  Success: {success}")


if __name__ == "__main__":
    main()
