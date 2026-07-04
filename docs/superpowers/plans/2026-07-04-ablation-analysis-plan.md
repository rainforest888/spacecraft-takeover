# Ablation & Failure Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three-tier analysis of SAC v4 spacecraft takeover model — failure case dissection, observation signal ablation, and mass inference validation.

**Architecture:** Three standalone analysis scripts (A, B, C) sharing the same env/agent imports. Tier A is pure data analysis (no training). Tier C uses frozen `best.pt` for eval experiments. Tier B trains lightweight variants with ablation dims zeroed in observation. A prerequisite step fixes the fuel mass range to realistic 100–150 kg.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, PyTorch 2.12, MuJoCo 3.6, SciPy

## Global Constraints

- Fuel mass range: FUEL_MASS_MIN=100, FUEL_MASS_MAX=150 (realistic propellant)
- Observation space: 10-dim [sigma(3), omega(3), self_fuel(1), omega_dot(1), inertia_response(1), att_err(1)]
- Ablation indices: omega_dot=7, inertia_response=8, att_err=9
- SAC v4 hyperparameters: fixed_alpha=0.2, hidden_dim=256, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005, batch_size=256
- Conda env: `spacraft` at `G:\Conda\envs\spacraft\python.exe`
- Training seeds: 42, 43, 44 per variant (Tier B)
- Output directory: `outputs/analysis/`

---

### Task 0: Fix Fuel Mass Range (Prerequisite)

**Files:**
- Modify: `envs/spacecraft_env_v2.py:68-69`

**Interfaces:**
- Produces: `FUEL_MASS_MIN = 100.0`, `FUEL_MASS_MAX = 150.0`

- [ ] **Step 1: Update fuel mass constants**

In `envs/spacecraft_env_v2.py`, lines 68-69, change:

```python
FUEL_MASS_MIN = 100.0   # kg (realistic propellant range)
FUEL_MASS_MAX = 150.0   # kg
```

- [ ] **Step 2: Verify the change**

```bash
"G:\Conda\envs\spacraft\python.exe" -c "
import sys; sys.path.insert(0,'.')
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2
print(f'FUEL_MASS_MIN={SpacecraftTakeoverEnvV2.FUEL_MASS_MIN}')
print(f'FUEL_MASS_MAX={SpacecraftTakeoverEnvV2.FUEL_MASS_MAX}')
assert SpacecraftTakeoverEnvV2.FUEL_MASS_MIN == 100.0
assert SpacecraftTakeoverEnvV2.FUEL_MASS_MAX == 150.0
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add envs/spacecraft_env_v2.py
git commit -m "fix: set fuel mass range to realistic 100-150 kg"
```

---

### Task 1: Tier A — Failure Analysis Script

**Files:**
- Create: `scripts/analyze_failures.py`

**Interfaces:**
- Consumes: `outputs/logs/training_log.csv` (existing 500-episode log)
- Produces: 6 PNG figures in `outputs/analysis/`

- [ ] **Step 1: Create output directory and write the full analysis script**

```bash
mkdir -p outputs/analysis
```

```python
# scripts/analyze_failures.py
"""Tier A: Failure case analysis — parse training log, compute stats, generate figures."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs", "training_log.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
cols = ["episode","total_reward","episode_length","success","alpha",
        "actor_loss","critic_loss","dry_mass","est_mass","fuel_mass_init","phase_switched"]
data = {}
with open(LOG_PATH) as f:
    header = f.readline().strip().split(",")
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < len(cols):
            continue
        for i, c in enumerate(cols):
            data.setdefault(c, []).append(float(parts[i]))

for k in data:
    data[k] = np.array(data[k])

success_mask = data["success"] == 1
failure_mask = data["success"] == 0
episodes = data["episode"]

# ── A1: Initial conditions vs success ──────────────────────────────────
def bucket_dry_mass(val):
    if val < 450: return "400-450"
    elif val < 550: return "450-550"
    else: return "550-600"

def bucket_fuel(val):
    if val < 115: return "100-115"
    elif val < 135: return "115-135"
    else: return "135-150"

dry_buckets = [bucket_dry_mass(v) for v in data["dry_mass"]]
fuel_buckets = [bucket_fuel(v) for v in data["fuel_mass_init"]]
dry_labels = ["400-450", "450-550", "550-600"]
fuel_labels = ["100-115", "115-135", "135-150"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Dry mass
dry_rates = []
for dl in dry_labels:
    mask = np.array([d == dl for d in dry_buckets])
    if mask.sum() > 0:
        dry_rates.append(data["success"][mask].mean() * 100)
    else:
        dry_rates.append(0)
axes[0].bar(dry_labels, dry_rates, color=["#2ecc71","#3498db","#e74c3c"])
axes[0].set_title("Success Rate by Dry Mass")
axes[0].set_ylabel("Success Rate (%)")
axes[0].set_ylim(0, 100)
for i, v in enumerate(dry_rates):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")

# Fuel mass
fuel_rates = []
for fl in fuel_labels:
    mask = np.array([f == fl for f in fuel_buckets])
    if mask.sum() > 0:
        fuel_rates.append(data["success"][mask].mean() * 100)
    else:
        fuel_rates.append(0)
axes[1].bar(fuel_labels, fuel_rates, color=["#2ecc71","#f39c12","#e74c3c"])
axes[1].set_title("Success Rate by Initial Fuel Mass")
axes[1].set_ylabel("Success Rate (%)")
axes[1].set_ylim(0, 100)
for i, v in enumerate(fuel_rates):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a1_initial_conditions.png"), dpi=150)
plt.close()
print("A1: initial conditions chart saved")

# ── A2: Phase switch comparison ────────────────────────────────────────
switched_success = data["phase_switched"][success_mask].mean() * 100
switched_failure = data["phase_switched"][failure_mask].mean() * 100
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Success", "Failure"], [switched_success, switched_failure],
       color=["#2ecc71", "#e74c3c"])
ax.set_title("Phase Switch Rate: Success vs Failure")
ax.set_ylabel("Phase Switched (%)")
for i, v in enumerate([switched_success, switched_failure]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a2_phase_switch.png"), dpi=150)
plt.close()
print(f"A2: phase switch — success={switched_success:.1f}%, failure={switched_failure:.1f}%")

# ── A3: Episode length distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data["episode_length"][success_mask], bins=30, alpha=0.6, label="Success", color="#2ecc71")
ax.hist(data["episode_length"][failure_mask], bins=30, alpha=0.6, label="Failure", color="#e74c3c")
ax.set_title("Episode Length Distribution")
ax.set_xlabel("Steps")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a3_episode_length.png"), dpi=150)
plt.close()
print(f"A3: episode length — success mean={data['episode_length'][success_mask].mean():.0f}, failure mean={data['episode_length'][failure_mask].mean():.0f}")

# ── A4: Reward trajectory (smoothed) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
window = 50
for label, mask in [("Success", success_mask), ("Failure", failure_mask)]:
    # Use episode order within each group
    idx = np.where(mask)[0]
    rewards = data["total_reward"][mask]
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(rewards)), smoothed, label=label, linewidth=2)
ax.set_title("Smoothed Reward Trajectory (window=50)")
ax.set_xlabel("Episode (within group)")
ax.set_ylabel("Total Reward")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a4_reward_trajectory.png"), dpi=150)
plt.close()
print("A4: reward trajectory chart saved")

# ── A5: Correlation heatmap ────────────────────────────────────────────
corr_cols = ["total_reward","episode_length","success","actor_loss","critic_loss",
             "dry_mass","est_mass","fuel_mass_init","phase_switched"]
corr_data = np.column_stack([data[c] for c in corr_cols])
corr = np.corrcoef(corr_data.T)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr_cols, fontsize=8)
ax.set_title("Feature Correlation Matrix")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a5_correlation_heatmap.png"), dpi=150)
plt.close()
print("A5: correlation heatmap saved")

# ── A6: Training curve ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Success rate over time
window = 20
success_series = np.array([data["success"][:i+1].mean() * 100 for i in range(len(episodes))])
if len(success_series) > window:
    smoothed = np.convolve(success_series, np.ones(window)/window, mode="valid")
    axes[0].plot(range(window-1, len(success_series)), smoothed, color="#3498db")
axes[0].set_title(f"Success Rate (smoothed, w={window})")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Success Rate (%)")
axes[0].set_ylim(0, 100)
axes[0].axhline(y=75.8, color="gray", linestyle="--", label="Final 75.8%")
axes[0].legend()

# Reward over time
rewards = data["total_reward"]
if len(rewards) > window:
    smoothed_r = np.convolve(rewards, np.ones(window)/window, mode="valid")
    axes[1].plot(range(window-1, len(rewards)), smoothed_r, color="#e74c3c")
axes[1].set_title("Average Reward (smoothed)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Total Reward")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "a6_training_curve.png"), dpi=150)
plt.close()
print("A6: training curve saved")

# ── Summary stats ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("FAILURE ANALYSIS SUMMARY")
print("="*60)
print(f"Total episodes: {len(episodes)}")
print(f"Success: {success_mask.sum()} ({success_mask.mean()*100:.1f}%)")
print(f"Failure: {failure_mask.sum()} ({failure_mask.mean()*100:.1f}%)")
print(f"\nSuccess episodes:")
print(f"  Mean dry_mass: {data['dry_mass'][success_mask].mean():.0f} kg")
print(f"  Mean fuel_mass_init: {data['fuel_mass_init'][success_mask].mean():.0f} kg")
print(f"  Mean episode_length: {data['episode_length'][success_mask].mean():.0f} steps")
print(f"  Phase switched: {data['phase_switched'][success_mask].mean()*100:.1f}%")
print(f"  Mean reward: {data['total_reward'][success_mask].mean():.1f}")
print(f"\nFailure episodes:")
print(f"  Mean dry_mass: {data['dry_mass'][failure_mask].mean():.0f} kg")
print(f"  Mean fuel_mass_init: {data['fuel_mass_init'][failure_mask].mean():.0f} kg")
print(f"  Mean episode_length: {data['episode_length'][failure_mask].mean():.0f} steps")
print(f"  Phase switched: {data['phase_switched'][failure_mask].mean()*100:.1f}%")
print(f"  Mean reward: {data['total_reward'][failure_mask].mean():.1f}")
print(f"\nCorrelation with success:")
for c in ["dry_mass", "fuel_mass_init", "phase_switched", "episode_length"]:
    cc = np.corrcoef(data[c], data["success"])[0, 1]
    print(f"  {c}: {cc:.4f}")
```

- [ ] **Step 2: Run the analysis script**

```bash
"G:\Conda\envs\spacraft\python.exe" scripts/analyze_failures.py
```

Expected: 6 PNG files created in `outputs/analysis/`, summary stats printed to console.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_failures.py outputs/analysis/
git commit -m "feat: Tier A — failure case analysis script and figures"
```

---

### Task 2: Tier C — Mass Inference Validation

**Files:**
- Create: `scripts/validate_mass_inference.py`

**Interfaces:**
- Consumes: `outputs/checkpoints/best.pt`, `envs/spacecraft_env_v2.py`, `algorithms/sac_agent.py`
- Produces: 3 PNG figures in `outputs/analysis/`

- [ ] **Step 1: Write the validation script**

```python
# scripts/validate_mass_inference.py
"""Tier C: Mass inference validation — sweep, perturbation, oracle."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2
from algorithms.sac_agent import SACAgent

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "best.pt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

OBS_DIM, ACT_DIM = 10, 3
MAX_STEPS = 600

def load_agent():
    agent = SACAgent(obs_dim=OBS_DIM, action_dim=ACT_DIM, hidden_dim=256,
                     actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    agent.load(CHECKPOINT)
    agent.actor.eval()
    return agent

def run_episode(env, agent, perturb_inertia=False):
    """Run one eval episode. Returns (success, episode_length, total_reward)."""
    obs, info = env.reset()
    total_r = 0.0
    for step in range(MAX_STEPS):
        if perturb_inertia:
            # Replace inertia_response with random noise (same ~0.5-0.64 range)
            obs[8] = np.random.uniform(0.3, 0.7)
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        if terminated or truncated:
            break
    success = info.get("target_fuel", 1.0) <= 0.0
    return success, step + 1, total_r

# ═══════════════════════════════════════════════════════════════════════
# C1: Mass Sweep — frozen model, vary dry_mass × fuel_mass
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("C1: Mass Sweep")
print("=" * 60)

dry_masses = [400.0, 500.0, 600.0]
fuel_masses = [100.0, 116.0, 133.0, 150.0]
N_EVAL = 20

results_c1 = np.zeros((len(dry_masses), len(fuel_masses)))
results_c1_len = np.zeros((len(dry_masses), len(fuel_masses)))

agent = load_agent()

for i, dm in enumerate(dry_masses):
    for j, fm in enumerate(fuel_masses):
        successes = 0
        lengths = []
        for ep in range(N_EVAL):
            env = SpacecraftTakeoverEnvV2(max_steps=MAX_STEPS)
            # Override mass ranges to fixed values
            env.DRY_MASS_MIN = dm
            env.DRY_MASS_MAX = dm
            env.FUEL_MASS_MIN = fm
            env.FUEL_MASS_MAX = fm
            success, length, _ = run_episode(env, agent)
            successes += int(success)
            lengths.append(length)
            env.close()
        results_c1[i, j] = successes / N_EVAL * 100
        results_c1_len[i, j] = np.mean(lengths)
        print(f"  dry={dm:.0f} fuel={fm:.0f}: {successes}/{N_EVAL} ({results_c1[i,j]:.1f}%)")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(results_c1, cmap="RdYlGn", vmin=0, vmax=100)
for i in range(len(dry_masses)):
    for j in range(len(fuel_masses)):
        ax.text(j, i, f"{results_c1[i,j]:.0f}%", ha="center", va="center", fontsize=13, fontweight="bold")
ax.set_xticks(range(len(fuel_masses)))
ax.set_yticks(range(len(dry_masses)))
ax.set_xticklabels([f"{f:.0f}" for f in fuel_masses])
ax.set_yticklabels([f"{d:.0f}" for d in dry_masses])
ax.set_xlabel("Initial Fuel Mass (kg)")
ax.set_ylabel("Dry Mass (kg)")
ax.set_title("Success Rate Heatmap: Frozen Model vs Mass Conditions")
plt.colorbar(im, ax=ax, label="Success Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "c1_mass_sweep_heatmap.png"), dpi=150)
plt.close()
print("C1 heatmap saved\n")

# ═══════════════════════════════════════════════════════════════════════
# C2: Signal Perturbation — shuffle inertia_response
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("C2: Signal Perturbation")
print("=" * 60)

N_PERTURB = 100
baseline_success = 0
perturb_success = 0

for ep in range(N_PERTURB):
    env = SpacecraftTakeoverEnvV2(max_steps=MAX_STEPS)
    s, _, _ = run_episode(env, agent, perturb_inertia=False)
    baseline_success += int(s)
    env.close()

for ep in range(N_PERTURB):
    env = SpacecraftTakeoverEnvV2(max_steps=MAX_STEPS)
    s, _, _ = run_episode(env, agent, perturb_inertia=True)
    perturb_success += int(s)
    env.close()

baseline_rate = baseline_success / N_PERTURB * 100
perturb_rate = perturb_success / N_PERTURB * 100
print(f"  Baseline (clean signal):  {baseline_success}/{N_PERTURB} ({baseline_rate:.1f}%)")
print(f"  Perturbed (noisy signal): {perturb_success}/{N_PERTURB} ({perturb_rate:.1f}%)")
print(f"  Drop: {baseline_rate - perturb_rate:.1f} pp")

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["Baseline", "Perturbed\n(inertia_response noise)"],
              [baseline_rate, perturb_rate],
              color=["#2ecc71", "#e74c3c"])
ax.set_ylabel("Success Rate (%)")
ax.set_title("C2: Signal Perturbation Test")
ax.set_ylim(0, 100)
for bar, v in zip(bars, [baseline_rate, perturb_rate]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "c2_perturbation_comparison.png"), dpi=150)
plt.close()
print("C2 chart saved\n")

# ═══════════════════════════════════════════════════════════════════════
# C3: Mass Oracle — train a "cheating" model with direct mass observation
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("C3: Mass Oracle (200 episodes)")
print("=" * 60)

# Quick check if oracle already exists
ORACLE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "oracle.pt")

if os.path.exists(ORACLE_PATH):
    print("  Oracle checkpoint exists, loading...")
    oracle_agent = SACAgent(obs_dim=12, action_dim=ACT_DIM, hidden_dim=256,
                            actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    oracle_agent.load(ORACLE_PATH)
else:
    print("  Training oracle model (12-dim obs with mass info)...")
    from algorithms.sac_agent import ReplayBuffer

    # We need a wrapper env that adds mass to observation
    class OracleEnv(SpacecraftTakeoverEnvV2):
        """Environment with direct mass observation for oracle upper bound."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Override observation space: add dry_mass and fuel_mass (normalized)
            obs_high = np.array(
                [np.inf]*6 + [1.0, np.inf, 1.0, np.inf, 1.0, 1.0],
                dtype=np.float32)
            obs_low = np.array(
                [-np.inf]*6 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32)
            self.observation_space = type(self.observation_space)(low=obs_low, high=obs_high, dtype=np.float32)

        def _get_obs(self):
            base = super()._get_obs()
            # Add normalized dry_mass and fuel_mass
            dry_norm = (self._dry_mass - 400.0) / 200.0   # normalize to ~[0,1]
            fuel_norm = self._fuel_mass / 150.0            # normalize to ~[0,1]
            return np.append(base, [dry_norm, fuel_norm]).astype(np.float32)

    oracle_agent = SACAgent(obs_dim=12, action_dim=ACT_DIM, hidden_dim=256,
                            actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    buffer = ReplayBuffer(100000, 12, ACT_DIM)

    env = OracleEnv(max_steps=MAX_STEPS)
    success_count = 0
    for ep in range(200):
        obs, info = env.reset()
        ep_r = 0.0
        for step in range(MAX_STEPS):
            action = oracle_agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            ep_r += reward
            if len(buffer) >= 256:
                oracle_agent.update(buffer.sample(256))
            if terminated or truncated:
                break
        if info.get("target_fuel", 1.0) <= 0.0:
            success_count += 1
        if (ep + 1) % 20 == 0:
            print(f"    Oracle ep {ep+1}/200 — success rate: {success_count/(ep+1)*100:.1f}%")
    env.close()
    oracle_agent.save(ORACLE_PATH)
    print(f"  Oracle trained: {success_count}/200 = {success_count/2:.1f}%")

# Evaluate oracle
oracle_success = 0
for ep in range(50):
    env = OracleEnv(max_steps=MAX_STEPS)
    obs, info = env.reset()
    for step in range(MAX_STEPS):
        action = oracle_agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    if info.get("target_fuel", 1.0) <= 0.0:
        oracle_success += 1
    env.close()
oracle_rate = oracle_success / 50 * 100
print(f"  Oracle eval (50 eps): {oracle_success}/50 = {oracle_rate:.1f}%")

# Baseline for reference
baseline_rate_c3 = baseline_rate  # from C2
print(f"  Baseline (blind): {baseline_rate_c3:.1f}%")
print(f"  Oracle (cheat):   {oracle_rate:.1f}%")
print(f"  Gap: {oracle_rate - baseline_rate_c3:.1f} pp")

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["Baseline\n(no mass info)", "Oracle\n(direct mass obs)"],
              [baseline_rate_c3, oracle_rate],
              color=["#3498db", "#f39c12"])
ax.set_ylabel("Success Rate (%)")
ax.set_title("C3: Mass Oracle Upper Bound")
ax.set_ylim(0, 100)
for bar, v in zip(bars, [baseline_rate_c3, oracle_rate]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "c3_oracle_gap.png"), dpi=150)
plt.close()
print("C3 chart saved\n")
print("Tier C complete.")
```

- [ ] **Step 2: Run C1+C2 (fast, uses frozen model)**

```bash
"G:\Conda\envs\spacraft\python.exe" scripts/validate_mass_inference.py
```

Expected: C1 mass sweep heatmap, C2 perturbation comparison, C3 oracle training + comparison. Approx 30 min for C3 training.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_mass_inference.py outputs/analysis/ outputs/checkpoints/oracle.pt
git commit -m "feat: Tier C — mass inference validation (sweep, perturbation, oracle)"
```

---

### Task 3: Add `ablated_dims` to Environment

**Files:**
- Modify: `envs/spacecraft_env_v2.py` — `__init__` and `_get_obs`

**Interfaces:**
- Consumes: (none — modifies existing class)
- Produces: `SpacecraftTakeoverEnvV2(ablated_dims: list[int] = None)`

- [ ] **Step 1: Add parameter and logic**

In `spacecraft_env_v2.py`, modify `__init__` signature (line 80):

```python
def __init__(self, render_mode=None, max_steps=600, ablated_dims=None):
    super().__init__()
    self.render_mode = render_mode
    self.max_steps = max_steps
    self.ablated_dims = ablated_dims or []  # list of observation indices to zero out
```

In `_get_obs` (line 359), add at the end of the function, before the return:

```python
def _get_obs(self) -> np.ndarray:
    sigma = self._get_mrp()
    omega = self.data.qvel[0:3].copy()
    att_err = float(np.linalg.norm(sigma))

    omega_dot = min(1.0, float(np.linalg.norm(self._alpha)) / 2.0)
    inertia_response = min(1.0, self._mass_response / 0.15)

    obs = np.array([
        sigma[0], sigma[1], sigma[2],
        omega[0], omega[1], omega[2],
        float(np.clip(self._self_fuel, 0.0, 1.0)),
        omega_dot,
        inertia_response,
        att_err,
    ], dtype=np.float32)

    # Zero out ablated dimensions
    for dim in self.ablated_dims:
        obs[dim] = 0.0

    return obs
```

- [ ] **Step 2: Quick smoke test**

```bash
"G:\Conda\envs\spacraft\python.exe" -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2

# Test full observation
env = SpacecraftTakeoverEnvV2(max_steps=10)
obs, _ = env.reset()
print(f'Full obs: {obs}')
assert obs.shape == (10,), f'Expected (10,), got {obs.shape}'

# Test ablated
env2 = SpacecraftTakeoverEnvV2(max_steps=10, ablated_dims=[7, 8])
obs2, _ = env2.reset()
print(f'Ablated obs (dims 7,8): {obs2}')
assert obs2[7] == 0.0, f'dim 7 should be 0, got {obs2[7]}'
assert obs2[8] == 0.0, f'dim 8 should be 0, got {obs2[8]}'
assert obs2[6] != 0.0, f'dim 6 should NOT be 0, got {obs2[6]}'
print('OK')
env.close()
env2.close()
"
```

- [ ] **Step 3: Commit**

```bash
git add envs/spacecraft_env_v2.py
git commit -m "feat: add ablated_dims parameter for observation ablation studies"
```

---

### Task 4: Tier B — Observation Ablation Training

**Files:**
- Create: `scripts/run_ablation.py`

**Interfaces:**
- Consumes: `envs/spacecraft_env_v2.py` (with ablated_dims), `algorithms/sac_agent.py`
- Produces: ablation training logs in `outputs/logs/`, 3 PNG figures in `outputs/analysis/`

- [ ] **Step 1: Write the ablation training script**

```python
# scripts/run_ablation.py
"""Tier B: Observation signal ablation — train variants with dims zeroed out."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2
from algorithms.sac_agent import SACAgent, ReplayBuffer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "analysis")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

OBS_DIM, ACT_DIM = 10, 3
MAX_STEPS = 600
EPISODES = 200
SEEDS = [42, 43, 44]

VARIANTS = {
    "Full":       {"ablated": [],           "label": "B0: Full (baseline)", "color": "#2ecc71"},
    "No-ω_dot":   {"ablated": [7],           "label": "B1: No-ω_dot",      "color": "#e67e22"},
    "No-IR":      {"ablated": [8],           "label": "B2: No-IR",          "color": "#e74c3c"},
    "No-AE":      {"ablated": [9],           "label": "B3: No-AttErr",     "color": "#9b59b6"},
    "No-mass":    {"ablated": [7, 8],         "label": "B4: No-mass-signals","color": "#c0392b"},
}

def train_variant(name, ablated_dims, seed):
    """Train one variant for EPISODES episodes. Returns list of success per episode."""
    env = SpacecraftTakeoverEnvV2(max_steps=MAX_STEPS, ablated_dims=ablated_dims)
    agent = SACAgent(obs_dim=OBS_DIM, action_dim=ACT_DIM, hidden_dim=256,
                     actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    buffer = ReplayBuffer(100000, OBS_DIM, ACT_DIM)

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    success_per_ep = []
    success_count = 0
    log_path = os.path.join(LOG_DIR, f"ablation_{name}_seed{seed}.csv")
    log_file = open(log_path, "w")
    log_file.write("episode,total_reward,episode_length,success,phase_switched,dry_mass,fuel_mass_init\n")
    log_file.flush()

    for episode in range(EPISODES):
        obs, info = env.reset()
        ep_r, step = 0.0, 0
        switched = False

        for step in range(MAX_STEPS):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            ep_r += reward
            if info.get("phase_switched", False):
                switched = True
            if len(buffer) >= 256:
                agent.update(buffer.sample(256))
            if terminated or truncated:
                break

        success = info.get("target_fuel", 1.0) <= 0.0
        if success:
            success_count += 1
        success_per_ep.append(success)

        log_file.write(f"{episode},{ep_r:.4f},{step+1},{int(success)},{int(switched)},"
                       f"{info.get('dry_mass', 0):.0f},{info.get('fuel_mass', 0):.0f}\n")
        log_file.flush()

    log_file.close()
    env.close()
    final_rate = success_count / EPISODES * 100
    print(f"  [{name}] seed={seed}: {success_count}/{EPISODES} ({final_rate:.1f}%)")
    return success_per_ep, final_rate

# Run all variants x seeds
print("=" * 60)
print(f"Tier B: Observation Ablation ({EPISODES} episodes × {len(SEEDS)} seeds)")
print("=" * 60)

all_results = {}
for variant_name, cfg in VARIANTS.items():
    print(f"\n--- {cfg['label']} ---")
    seed_rates = []
    for seed in SEEDS:
        success_list, final_rate = train_variant(variant_name, cfg["ablated"], seed)
        seed_rates.append(final_rate)
    all_results[variant_name] = {
        "rates": seed_rates,
        "mean": np.mean(seed_rates),
        "std": np.std(seed_rates),
        "label": cfg["label"],
        "color": cfg["color"],
    }

# ── B1: Success rate comparison bar chart ──────────────────────────────
print("\n\nGenerating comparison figures...")
fig, ax = plt.subplots(figsize=(10, 6))
names = list(VARIANTS.keys())
means = [all_results[n]["mean"] for n in names]
stds = [all_results[n]["std"] for n in names]
colors = [VARIANTS[n]["color"] for n in names]
labels = [VARIANTS[n]["label"] for n in names]

bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
ax.set_ylabel("Success Rate (%)")
ax.set_title(f"B1: Observation Signal Ablation ({EPISODES} eps × {len(SEEDS)} seeds)")
ax.set_ylim(0, 100)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 2,
            f"{m:.1f}%", ha="center", fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "b1_success_rate_comparison.png"), dpi=150)
plt.close()
print("B1: saved")

# ── B2: Contribution table ─────────────────────────────────────────────
baseline_mean = all_results["Full"]["mean"]
fig, ax = plt.subplots(figsize=(8, 5))
contributions = []
contrib_labels = []
for n in ["No-ω_dot", "No-IR", "No-AE", "No-mass"]:
    drop = baseline_mean - all_results[n]["mean"]
    contributions.append(drop)
    contrib_labels.append(VARIANTS[n]["label"])

bars = ax.barh(contrib_labels, contributions, color=["#e67e22","#e74c3c","#9b59b6","#c0392b"])
ax.set_xlabel("Success Rate Drop (pp)")
ax.set_title("B2: Per-Signal Contribution (higher = more important)")
for bar, v in zip(bars, contributions):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{v:.1f} pp", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "b2_contribution_chart.png"), dpi=150)
plt.close()
print("B2: saved")

# ── B3: Summary table ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")
table_data = []
for n in names:
    r = all_results[n]
    # Calculate drop from baseline
    if n == "Full":
        drop_str = "—"
    else:
        drop_str = f"-{baseline_mean - r['mean']:.1f} pp"
    table_data.append([r["label"], f"{r['mean']:.1f}%", f"±{r['std']:.1f}%", drop_str])

table = ax.table(cellText=table_data,
                 colLabels=["Variant", "Success Rate", "Std Dev", "vs Baseline"],
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
ax.set_title("B3: Ablation Results Summary", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "b3_results_table.png"), dpi=150)
plt.close()
print("B3: saved")

# Console summary
print("\n" + "=" * 60)
print("ABLATION RESULTS SUMMARY")
print("=" * 60)
print(f"{'Variant':<25} {'Mean':>8} {'Std':>8} {'Drop':>8}")
print("-" * 50)
for n in names:
    r = all_results[n]
    drop = baseline_mean - r["mean"] if n != "Full" else 0
    print(f"{r['label']:<25} {r['mean']:>7.1f}% {r['std']:>7.1f}% {drop:>7.1f} pp")
print("-" * 50)
print(f"\nMost important signal: the one with largest drop when removed.")
print("Tier B complete.")
```

- [ ] **Step 2: Run ablation training**

```bash
"G:\Conda\envs\spacraft\python.exe" scripts/run_ablation.py
```

Expected: 5 variants × 3 seeds = 15 training runs × 200 episodes each. Approx 1.5 hours. Figures saved to `outputs/analysis/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ablation.py outputs/analysis/b*.png outputs/logs/ablation_*.csv
git commit -m "feat: Tier B — observation signal ablation study"
```

---

### Task 5: Final Summary and Cleanup

**Files:**
- Modify: `README.md` — add analysis results section

- [ ] **Step 1: Update README with analysis results**

Add a new section to README.md summarizing the key findings from all three tiers.

- [ ] **Step 2: Final commit**

```bash
git add README.md
git commit -m "docs: add ablation analysis results to README"
```
