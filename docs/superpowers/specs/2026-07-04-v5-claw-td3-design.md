# Spacecraft Takeover V5 — Claw Mechanism + TD3 + Efficiency-Oriented Design

**Date**: 2026-07-04
**Status**: Draft → Implementation
**Depends on**: V4 ablation analysis findings

---

## 1. Overview

V5 redesign based on ablation findings:
- **Finding**: Agent uses "stabilize + torque" strategy; mass signals unused; 92% baseline with att_err alone
- **Fix**: Make fuel efficiency the core objective; self_fuel becomes real constraint
- **Algorithm**: Switch from SAC (stochastic, entropy issues) to **TD3** (deterministic, stable)
- **Model**: Claw-type gripping mechanism replacing invisible weld

---

## 2. MuJoCo Model: Claw Mechanism

### 2.1 Architecture

```
service_sat (0.4m cube, 100kg)
  ├── claw_top    (0.10×0.70×0.10 m)  +X extending from top edge
  ├── claw_bottom (0.10×0.70×0.10 m)  +X extending from bottom edge
  ├── claw_left   (0.10×0.10×0.70 m)  +X extending from left edge
  └── claw_right  (0.10×0.10×0.70 m)  +X extending from right edge

target_sat (1.0×0.5×0.5 m, 400-600 kg + 100-150 kg fuel)
  → Enclosed within claw cage
  → weld constraint: relpose="1.0 0 0 1 0 0 0"
```

All claw arms are fixed geoms on service_sat body (no extra joints). The four arms form a rectangular cage extending 0.7m forward, enclosing the target spacecraft.

### 2.2 Visual Appearance

- Service sat: blue box + 4 gray claw arms
- Target sat inside the cage: red box + 2 blue solar panels

---

## 3. Environment: V5 Parameters

### 3.1 Critical Changes

| Parameter | V4 | V5 | Rationale |
|-----------|-----|-----|-----------|
| `W_SELF_BURN` | 1.0 | **5.0** | Self-burn penalty 5x stronger |
| `self_burn_rate` | 0.006 | **0.012** | Double self fuel consumption rate |
| `W_EFF` | — | **3.0** | New: efficiency reward weight |
| `MAX_ATT_ERR` | π | **π/2** | Tighter attitude constraint |
| `FUEL_K_MASS` | 2.0 | **3.0** | Faster target fuel depletion |

### 3.2 Reward Function

```
r = SCALE * (
    W_FUEL_BURN * fuel_burned_kg           ← + reward for depleting target
  - W_SELF_BURN * self_burn                ← − penalty for using own fuel (×5)
  - W_STEP                                 ← − per-step cost
  + W_EFF * fuel_burned_kg / (self_burn + 1e-6)  ← + efficiency bonus
)
```

### 3.3 Terminal Conditions

| Condition | Outcome | Reward |
|-----------|---------|--------|
| `target_fuel ≤ 0` | **Success** | +R_SUCCESS (can continue for attitude hold) |
| `self_fuel ≤ 0` | Failure | +R_FAIL |
| `att_err > π/2` | Failure | +R_FAIL |
| `step ≥ max_steps` | Truncated | (no bonus) |

`max_steps=600` is safety net only. The real constraint is `self_fuel`.

### 3.4 Target Strategies

Mix LQR + SMC + PID (all 3 families, randomly selected per episode). The V4 env only used LQR.

---

## 4. Algorithm: TD3

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| `actor_lr` | 1e-4 |
| `critic_lr` | 3e-4 |
| `gamma` | 0.99 |
| `tau` | 0.005 |
| `policy_noise` | 0.2 |
| `noise_clip` | 0.5 |
| `policy_delay` | 2 |
| `hidden_dim` | 256 |
| `batch_size` | 256 |
| `buffer_size` | 100000 |
| `exploration_noise` | 0.3 (decay to 0.05) |

### 4.2 Observation Space (unchanged, 10-dim)

```
[sigma(3), omega(3), self_fuel(1), omega_dot(1), inertia_response(1), att_err(1)]
```

---

## 5. Training Protocol

- 1000 episodes
- Save checkpoint every 200 episodes
- Evaluate every 50 episodes on deterministic policy
- Track: success rate, mean reward, self_fuel remaining, target fuel depletion time
