# Spacecraft Takeover — Ablation & Failure Analysis Design

**Date**: 2026-07-04
**Status**: Approved → Ready for implementation plan
**Depends on**: SAC v4 (75.8% success rate, fixed-alpha=0.2)

---

## 1. Overview

Three-tier analysis of the SAC v4 spacecraft takeover model:

- **A. Failure case analysis** — statistical dissection of the ~24% failure episodes from existing training logs
- **B. Observation signal ablation** — train variants with individual observation dimensions removed to quantify each signal's contribution
- **C. Mass inference validation** — controlled experiments to verify the agent truly learns mass from dynamics, not shortcuts

---

## 2. Tier A: Failure Case Analysis

### 2.1 Data Source

`outputs/logs/training_log.csv` — 500 episodes with columns:
`episode, total_reward, episode_length, success, alpha, actor_loss, critic_loss, dry_mass, est_mass, fuel_mass_init, phase_switched`

### 2.2 Analysis Dimensions

| Dimension | Method | Question |
|-----------|--------|----------|
| Initial conditions | Bucket by `dry_mass` (400/500/600) × `fuel_mass_init` (low/med/high), compute success rate per bucket | Are heavy+fuel-rich targets harder? |
| Phase switch | Compare `phase_switched` ratio in success vs failure episodes | Do failures fail to trigger the phase switch? |
| Episode length | Distribution comparison (success vs failure) | Are failures "fast crashes" or "600-step timeouts"? |
| Reward trajectory | Rolling-window reward curves for success/failure groups | At what step does divergence emerge? |
| Feature correlation | Point-biserial correlation of all numeric features with `success` | Which logged signals actually correlate with outcome? |
| Training dynamics | Success rate over episodes, smoothed; reward vs episode | Is performance still improving or plateaued? |

### 2.3 Output

- Script: `scripts/analyze_failures.py`
- Figures saved to `outputs/analysis/`:
  - `a1_initial_conditions.png` — grouped bar chart
  - `a2_phase_switch.png` — stacked bar
  - `a3_episode_length.png` — histogram (success vs failure)
  - `a4_reward_trajectory.png` — smoothed line plot
  - `a5_correlation_heatmap.png` — correlation matrix
  - `a6_training_curve.png` — success rate + reward over episodes

---

## 3. Tier B: Observation Signal Ablation

### 3.1 Observation Space (10-dim)

```
[sigma(3), omega(3), self_fuel(1), omega_dot(1), inertia_response(1), att_err(1)]
```

### 3.2 Variants

| Variant | Ablated Signal(s) | Hypothesis |
|---------|-------------------|------------|
| B0: Full | None (baseline) | Current 75.8% |
| B1: No-ω̇ | `omega_dot` (index 7) | Removing angular accel should degrade — agent loses direct dynamics feedback |
| B2: No-IR | `inertia_response` (index 8) | Removing inertia response should hurt mass inference — bigger impact than ω̇ |
| B3: No-AE | `att_err` (index 9) | Removing attitude error — agent may lose stabilization signal |
| B4: No-mass | `omega_dot` + `inertia_response` (index 7,8) | Both mass-related signals gone — largest expected degradation |

### 3.3 Implementation

Add `ablated_dims: list[int]` parameter to `SpacecraftTakeoverEnvV2`. In `_get_obs()`, zero out the specified indices after computing the full observation. No other code changes needed.

### 3.4 Protocol

- 200 episodes per variant
- 3 seeds each (42, 43, 44) → report mean ± std
- Same hyperparameters as v4 (fixed_alpha=0.2, etc.)
- Compare: success rate, mean reward, phase switch rate, episode length

### 3.5 Output

- Script: `scripts/run_ablation.py` (trains all variants sequentially)
- Figures:
  - `b1_success_rate_comparison.png` — grouped bar with error bars
  - `b2_reward_curves.png` — smoothed learning curves per variant
  - `b3_contribution_table.png` — heatmap/table of per-signal contribution

---

## 4. Tier C: Mass Inference Validation

### 4.1 Experiment C1 — Mass Sweep

- Freeze `best.pt` (no training)
- Sweep over 12 mass conditions: `dry_mass ∈ {400, 500, 600}` × `fuel_mass ∈ {100, 116, 133, 150}`
- 20 episodes per condition = 240 total eval episodes
- Note: fuel range (100–150 kg) matches realistic spacecraft propellant mass; requires updating env `FUEL_MASS_MIN=100, FUEL_MASS_MAX=150`
- Hypothesis: success rate should be higher for lighter/less-fuel targets — confirms agent adapts to mass

### 4.2 Experiment C2 — Signal Perturbation

- Freeze `best.pt`
- During eval, replace `inertia_response` with random noise (same distribution mean/std)
- 100 eval episodes with perturbed signal vs 100 baseline
- Hypothesis: success rate drops significantly → agent genuinely relies on this signal for mass inference

### 4.3 Experiment C3 — Mass Oracle Upper Bound

- Train a small "cheating" model (200 episodes) that has direct access to `target_fuel` and `dry_mass` in observation
- Compare its success rate to the blind baseline
- Hypothesis: quantifies the "mass uncertainty penalty" — how much performance is sacrificed by not observing mass directly

### 4.4 Output

- Script: `scripts/validate_mass_inference.py`
- Figures:
  - `c1_mass_sweep_heatmap.png` — success rate heatmap (dry_mass × fuel_mass)
  - `c2_perturbation_comparison.png` — bar chart with error bars
  - `c3_oracle_gap.png` — bar chart showing baseline vs oracle gap

---

## 5. Execution Order

### 5.1 Prerequisite: Fix Fuel Mass Range

Update `SpacecraftTakeoverEnvV2` to realistic propellant range:
- `FUEL_MASS_MIN`: 50 → **100** kg
- `FUEL_MASS_MAX`: 200 → **150** kg

### 5.2 Analysis Order

```
A (existing data, ~5 min) → C (existing model, ~30 min) → B (train variants, ~1.5 hr)
```

A and C require zero training — they run on existing data and the frozen best.pt. B is the most compute-intensive and runs last.

---

## 6. File Map

| File | Purpose |
|------|---------|
| `scripts/analyze_failures.py` | Tier A: parse log, compute stats, generate figures |
| `envs/spacecraft_env_v2.py` | Add `ablated_dims` parameter for Tier B |
| `scripts/run_ablation.py` | Tier B: train all variants, collect results, plot |
| `scripts/validate_mass_inference.py` | Tier C: mass sweep, perturbation test, oracle training |
| `outputs/analysis/` | All output figures |
| `outputs/logs/ablation_*.csv` | Ablation training logs |

---

## 7. Self-Review

- **Placeholder scan**: No TBD/TODO — all dimensions, metrics, and outputs are specified.
- **Internal consistency**: All three tiers use consistent metrics (success rate, episode reward, phase switch rate). Tier B and C share the same observation space definition.
- **Scope**: Three tiers, each with clear deliverables. No scope creep. One implementation plan can cover all three.
- **Ambiguity**: Signal indices (7, 8, 9) are explicit. Seed values (42, 43, 44) specified. Experiment counts explicit.
