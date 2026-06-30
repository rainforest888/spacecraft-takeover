# Spacecraft Takeover Control — Design Specification

**Project**: 空间非合作航天器的智能姿态接管控制研究  
**Date**: 2026-06-30  
**Status**: Approved → Ready for implementation plan

---

## 1. Overview

Build a MuJoCo-based simulation environment and TD3-LQR two-stage control pipeline for attitude takeover of a non-cooperative spacecraft. A small service spacecraft (我方, ~100 kg) has captured a large non-cooperative target (非合作目标, ~500 kg). The combined-body system must be stabilized through two phases:

1. **Weakening phase (弱化)** — TD3 reinforcement learning agent outputs torque to induce the target to burn its fuel resisting, depleting its maneuvering capability.
2. **Takeover phase (接管)** — Once the target is weakened (observable via torque output decay), LQR controller smoothly takes over to stabilize the combined body's attitude.

## 2. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MuJoCo built-in geometry first, SolidWorks STL later | Unblocks algorithm development; model swap requires zero code changes |
| Combined body via weld constraint, starting from "already captured" state | Skips rendezvous/capture dynamics — not in scope |
| 11-dim observation (no target fuel reading) | Target fuel is unobservable in reality; inferred from torque proxy |
| LEO gravity gradient torque as sole environmental disturbance | Dominant perturbation; others negligible at control timescales |
| CPU-only compatible, GPU-accelerated training | RTX 5060 8 GB VRAM, PyTorch CUDA build required |
| Gymnasium (gym) interface | Standard RL environment interface, reusable across algorithms |
| One-week MVP scope | Full pipeline: env → train → eval → video |

## 3. Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Physics engine | MuJoCo | ≥ 3.0 |
| Deep learning | PyTorch (CUDA) | latest stable |
| RL environment | Gymnasium | latest |
| Numerical | NumPy, SciPy | latest |
| Visualization | MuJoCo renderer + matplotlib | — |
| CAD | SolidWorks → STL → MJCF | user-provided |
| Package manager | Anaconda + pip | — |

## 4. Project Structure

```
spacecraft-takeover/
├── models/
│   ├── meshes/              # STL files (user-provided, later)
│   └── mjcf/
│       └── combo_body.xml   # MuJoCo model (built-in geometry)
├── envs/
│   ├── __init__.py
│   ├── spacecraft_env.py    # Main Gym env
│   └── dynamics.py          # Euler dynamics, fuel model, gravity gradient
├── algorithms/
│   ├── __init__.py
│   ├── td3_agent.py         # TD3 implementation
│   ├── lqr_controller.py    # LQR with Riccati solver
│   ├── switch_manager.py    # State machine for phase switching
│   └── target_controllers.py # Target's adversarial policies (PID/SMC/LQR)
├── scripts/
│   ├── train.py             # Offline TD3 training
│   ├── eval.py              # Evaluation + metrics
│   ├── demo.py              # MuJoCo render + video recording
│   └── compare.py           # Baseline comparisons
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── videos/
├── tests/
│   ├── test_env.py
│   ├── test_td3.py
│   └── test_switch.py
├── requirements.txt
└── README.md
```

## 5. MuJoCo Environment Design

### 5.1 Scene

Two spacecraft bodies connected via `<weld>` equality constraint:

- **Service sat**: 1×1×1 m, 100 kg, diaginertia ~16.7
- **Target sat**: 3×2×2 m, 500 kg, diaginertia ~500
- 8 thruster sites per body (4 faces × 2 per face), producing torque couples
- Zero gravity (`gravity="0 0 0"`)
- Timestep: 0.002 s (500 Hz), control runs at 60 Hz (substeps skipped for RL)

### 5.2 Gym Interface

```python
# Observation space (11-dim, Box)
obs = [
    σ₁, σ₂, σ₃,              # MRP attitude error (3)
    ω₁, ω₂, ω₃,              # Combined body angular velocity (3)
    f_self,                  # Self fuel fraction [0, 1]
    τ_target_mag,            # Estimated target torque magnitude
    τ_target_mag_ma,         # Moving average of target torque (window=30)
    Δτ_target,               # Torque trend (+increasing / -decreasing)
    t_elapsed                # Normalized elapsed time [0, 1]
]

# Action space (3-dim, Box continuous)
action = [τ₁, τ₂, τ₃]        # Normalized [-1, 1], scaled to max torque
```

### 5.3 Reward Function

```python
# Per-step reward
reward = (
    w1 * fuel_consumed_by_target       # w1 = 10.0  Primary objective
    - w2 * fuel_consumed_by_self       # w2 = 5.0   Self-preservation
    - w3 * attitude_error_norm         # w3 = 2.0   Tumble prevention
    - w4 * angular_velocity_norm       # w4 = 1.0   Soft damping
    + w5 * burst_bonus                 # w5 = 3.0   Short-term depletion burst
    + w6 * weakening_bonus             # w6 = 2.0   Detecting target decay
)

# Terminal rewards
if target_fuel <= 0:     reward += +100    # Success
elif self_fuel <= 0:     reward += -100    # Self-exhausted
elif attitude_error > MAX: reward += -100  # Tumble failure
```

Where:
- `burst_bonus = max(0, recent_target_burn - baseline_burn_rate * window)`
- `weakening_bonus = max(0, past_target_torque_mag - current_target_torque_mag)`

### 5.4 Fuel Model

```
fuel(t+1) = fuel(t) - Δt * ||τ|| * k_consumption
```
Linear with torque magnitude. Real fuel depletion time tuned via `k_consumption` to yield ~200-400 step episodes.

### 5.5 Target Adversarial Strategies

Each episode randomly samples one strategy from:
- **PID** × 3 variants (kp, ki, kd ranges)
- **SMC** × 3 variants (λ, η ranges)
- **LQR** × 4 variants (Q, R weight ranges)

Total: 10 distinct opponent profiles. Agent never sees which is active.

### 5.6 Environmental Disturbance

Gravity gradient torque (LEO, ~400 km circular orbit):
```
τ_gg = (3μ / R³) · (r̂ × I · r̂)
```
Where μ = 3.986×10¹⁴ m³/s², R = 6.8×10⁶ m, r̂ slowly rotates (orbital period ~93 min, effectively static per episode).

## 6. TD3 Algorithm Design

### 6.1 Architecture

```
Actor:  11(obs) → 256 → 256 → 3(action·tanh)
Critic: 14(obs+action) → 256 → 256 → 1(Q-value)
        Twin critics, take minimum to suppress overestimation

Dual 256-hidden-layer, well within RTX 5060 8 GB capacity.
```

### 6.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Actor learning rate | 1e-4 |
| Critic learning rate | 3e-4 |
| Replay buffer | 100,000 |
| Batch size | 256 |
| Target network τ | 0.005 |
| Exploration noise σ | 0.1 |
| Target policy noise σ | 0.2 (clipped ±0.5) |
| Policy delay | 2 |
| Discount γ | 0.99 |
| Episodes | 5,000–10,000 |
| Max steps/episode | 500 |

### 6.3 Training Flow

```
for episode in range(N_episodes):
    obs = env.reset()
    for step in range(500):
        action = actor(obs) + exploration_noise
        next_obs, reward, done, _ = env.step(action)
        buffer.store(obs, action, reward, next_obs, done)
        if buffer.size > batch_size:
            batch = buffer.sample(batch_size)
            update_critics(batch)
            if step % policy_delay == 0:
                update_actor(batch)
                soft_update_targets()
        if done: break
```

## 7. LQR Controller & Switch Manager

### 7.1 LQR Design

Linearized around target pointing equilibrium:
```
x = [σ₁, σ₂, σ₃, ω₁, ω₂, ω₃]ᵀ  (6-dim state)
u = [τ₁, τ₂, τ₃]ᵀ               (3-dim control)

Q = diag([50, 50, 50, 10, 10, 10])   # Attitude error > angular velocity
R = diag([1, 1, 1])                   # Unit fuel cost

AᵀP + PA - PBR⁻¹BᵀP + Q = 0  →  K = R⁻¹BᵀP  (via scipy.linalg.solve_continuous_are)
u = -K·x
```

### 7.2 Switch State Machine

```
WEAPENING → (target_torque_mag < 10% peak, sustained 30+ steps) → TRANSITION → LQR
```

**Smooth handover**: first 50 steps after switch, blend old/new control via exponential decay:
```
τ_output(t) = α(t)·τ_TD3 + (1-α(t))·τ_LQR
α(t) = max(0, 1 - t/50)
```

### 7.3 Transition Fallback

If attitude error > threshold at switch time, use proportional controller first to bring error down, then engage LQR.

### 7.4 Switch Criteria (all observable)

| Criterion | Condition |
|-----------|-----------|
| Primary | τ_target_mag_ma < 10% of observed peak, sustained ≥ 30 steps |
| Trend | τ_target decreasing AND < 5% peak |
| Desperation | self_fuel < 10% |

## 8. What's NOT in Scope (YAGNI)

- Rendezvous/capture dynamics (orbital approach, docking mechanism)
- Reaction wheel or CMG actuator modeling (torque-only abstraction)
- Sensor noise / state estimation filter (perfect observation)
- Communication delay or partial observability
- Multi-agent / swarm microsatellite scenario
- Hardware-in-the-loop testing
- Full orbital mechanics (only gravity gradient torque)
- Solar radiation pressure / atmospheric drag
- Flexible body or fuel slosh dynamics

## 9. SolidWorks → MuJoCo Workflow (User-Executed)

1. Model both satellites as single-part files in SolidWorks (meters)
2. Export STL with coordinate system info, < 5000 faces each
3. MeshLab decimate if needed
4. Place STL files in `models/meshes/`
5. Edit `models/mjcf/combo_body.xml`: replace `type="box"` with `type="mesh" mesh="service_sat|target_sat"`
6. Environment code requires zero changes

## 10. Success Criteria

- [ ] MuJoCo environment renders and runs with built-in geometry
- [ ] TD3 training converges (target depleted in >80% of test episodes)
- [ ] Switch manager correctly detects target weakening from torque proxy
- [ ] LQR takeover stabilizes combined body to <1° pointing error
- [ ] Full pipeline demo video rendered
- [ ] Training curves + comparison plots (vs PID-only, LQR-only baselines)
- [ ] Code committed and pushed to GitHub repo
