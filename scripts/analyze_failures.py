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

# ── A4: Reward trajectory (smoothed by group) ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
window = 50
for label, mask in [("Success", success_mask), ("Failure", failure_mask)]:
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
