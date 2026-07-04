# scripts/run_ablation.py
"""Tier B: Observation signal ablation — train variants with dims zeroed out."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2
from algorithms.sac_agent import SACAgent, ReplayBuffer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "analysis")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

OBS_DIM, ACT_DIM = 10, 3
MAX_STEPS = 600
EPISODES = 200
SEEDS = [42, 43, 44]

VARIANTS = {
    "Full":       {"ablated": [],           "label": "B0: Full (baseline)", "color": "#2ecc71"},
    "No-omega_dot":   {"ablated": [7],           "label": "B1: No-omega_dot",      "color": "#e67e22"},
    "No-IR":      {"ablated": [8],           "label": "B2: No-IR",          "color": "#e74c3c"},
    "No-AE":      {"ablated": [9],           "label": "B3: No-AttErr",     "color": "#9b59b6"},
    "No-mass":    {"ablated": [7, 8],         "label": "B4: No-mass-signals","color": "#c0392b"},
}

def train_variant(name, ablated_dims, seed):
    """Train one variant for EPISODES episodes."""
    env = SpacecraftTakeoverEnvV2(max_steps=MAX_STEPS, ablated_dims=ablated_dims)
    agent = SACAgent(obs_dim=OBS_DIM, action_dim=ACT_DIM, hidden_dim=256,
                     actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    buffer = ReplayBuffer(100000, OBS_DIM, ACT_DIM)

    np.random.seed(seed)
    torch.manual_seed(seed)

    success_count = 0
    log_path = os.path.join(LOG_DIR, f"ablation_{name}_seed{seed}.csv")
    log_file = open(log_path, "w")
    log_file.write("episode,total_reward,episode_length,success,phase_switched,dry_mass,fuel_mass_init\n")
    log_file.flush()

    for episode in range(EPISODES):
        obs, info = env.reset()
        ep_r = 0.0
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

        log_file.write(f"{episode},{ep_r:.4f},{step+1},{int(success)},{int(switched)},"
                       f"{info.get('dry_mass', 0):.0f},{info.get('fuel_mass', 0):.0f}\n")
        log_file.flush()

    log_file.close()
    env.close()
    final_rate = success_count / EPISODES * 100
    print(f"  [{name}] seed={seed}: {success_count}/{EPISODES} ({final_rate:.1f}%)")
    return final_rate

# Run all variants x seeds
print("=" * 60)
print(f"Tier B: Observation Ablation ({EPISODES} episodes x {len(SEEDS)} seeds)")
print("=" * 60)

all_results = {}
for variant_name, cfg in VARIANTS.items():
    print(f"\n--- {cfg['label']} ---")
    seed_rates = []
    for seed in SEEDS:
        final_rate = train_variant(variant_name, cfg["ablated"], seed)
        seed_rates.append(final_rate)
    all_results[variant_name] = {
        "rates": seed_rates,
        "mean": np.mean(seed_rates),
        "std": np.std(seed_rates),
        "label": cfg["label"],
        "color": cfg["color"],
    }
    print(f"  Mean: {all_results[variant_name]['mean']:.1f}% +/- {all_results[variant_name]['std']:.1f}%")

# ── B1: Success rate comparison bar chart ──────────────────────────────
print("\n\nGenerating comparison figures...")
fig, ax = plt.subplots(figsize=(12, 6))
names = list(VARIANTS.keys())
means = [all_results[n]["mean"] for n in names]
stds = [all_results[n]["std"] for n in names]
colors = [VARIANTS[n]["color"] for n in names]
labels = [VARIANTS[n]["label"] for n in names]

bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='white')
ax.set_ylabel("Success Rate (%)")
ax.set_title(f"B1: Observation Signal Ablation ({EPISODES} eps x {len(SEEDS)} seeds)")
ax.set_ylim(0, 100)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 2,
            f"{m:.1f}%", ha="center", fontweight="bold", fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "b1_success_rate_comparison.png"), dpi=150)
plt.close()
print("B1: saved")

# ── B2: Contribution chart ─────────────────────────────────────────────
baseline_mean = all_results["Full"]["mean"]
fig, ax = plt.subplots(figsize=(8, 5))
contributions = []
contrib_labels = []
contrib_colors = []
for n in ["No-omega_dot", "No-IR", "No-AE", "No-mass"]:
    drop = baseline_mean - all_results[n]["mean"]
    contributions.append(drop)
    contrib_labels.append(VARIANTS[n]["label"])
    contrib_colors.append(VARIANTS[n]["color"])

bars = ax.barh(contrib_labels, contributions, color=contrib_colors)
ax.set_xlabel("Success Rate Drop (pp)")
ax.set_title("B2: Per-Signal Contribution (higher drop = more important)")
for bar, v in zip(bars, contributions):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{v:.1f} pp", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "b2_contribution_chart.png"), dpi=150)
plt.close()
print("B2: saved")

# ── B3: Summary table ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")
table_data = []
for n in names:
    r = all_results[n]
    if n == "Full":
        drop_str = "—"
    else:
        drop_str = f"-{baseline_mean - r['mean']:.1f} pp"
    table_data.append([r["label"], f"{r['mean']:.1f}%", f"±{r['std']:.1f}%", drop_str])

table = ax.table(cellText=table_data,
                 colLabels=["Variant", "Success Rate", "Std Dev", "vs Baseline"],
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0 and row > 0:
        cell.set_text_props(ha="left")
ax.set_title("B3: Ablation Results Summary", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "b3_results_table.png"), dpi=150)
plt.close()
print("B3: saved")

# Console summary
print("\n" + "=" * 60)
print("ABLATION RESULTS SUMMARY")
print("=" * 60)
print(f"{'Variant':<28} {'Mean':>8} {'Std':>8} {'Drop':>8}")
print("-" * 55)
for n in names:
    r = all_results[n]
    drop = baseline_mean - r["mean"] if n != "Full" else 0
    print(f"{r['label']:<28} {r['mean']:>7.1f}% {r['std']:>7.1f}% {drop:>7.1f} pp")
print("-" * 55)
print(f"\nMost important signal: the one with largest drop when removed.")
print("Tier B complete.")
