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
            obs[8] = np.random.uniform(0.3, 0.7)
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        if terminated or truncated:
            break
    success = info.get("target_fuel", 1.0) <= 0.0
    return success, step + 1, total_r

# ═══════════════════════════════════════════════════════════════════════
# C1: Mass Sweep — frozen model, vary dry_mass x fuel_mass
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("C1: Mass Sweep")
print("=" * 60)

dry_masses = [400.0, 500.0, 600.0]
fuel_masses = [100.0, 116.0, 133.0, 150.0]
N_EVAL = 20

results_c1 = np.zeros((len(dry_masses), len(fuel_masses)))

agent = load_agent()

for i, dm in enumerate(dry_masses):
    for j, fm in enumerate(fuel_masses):
        successes = 0
        for ep in range(N_EVAL):
            env = SpacecraftTakeoverEnvV2(max_steps=MAX_STEPS)
            env.DRY_MASS_MIN = dm
            env.DRY_MASS_MAX = dm
            env.FUEL_MASS_MIN = fm
            env.FUEL_MASS_MAX = fm
            success, _, _ = run_episode(env, agent)
            successes += int(success)
            env.close()
        results_c1[i, j] = successes / N_EVAL * 100
        print(f"  dry={dm:.0f} fuel={fm:.0f}: {successes}/{N_EVAL} ({results_c1[i,j]:.1f}%)")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(results_c1, cmap="RdYlGn", vmin=0, vmax=100)
for i in range(len(dry_masses)):
    for j in range(len(fuel_masses)):
        ax.text(j, i, f"{results_c1[i,j]:.0f}%", ha="center", va="center",
                fontsize=13, fontweight="bold")
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

ORACLE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "oracle.pt")

if os.path.exists(ORACLE_PATH):
    print("  Oracle checkpoint exists, loading...")
    oracle_agent = SACAgent(obs_dim=12, action_dim=ACT_DIM, hidden_dim=256,
                            actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    oracle_agent.load(ORACLE_PATH)
else:
    print("  Training oracle model (12-dim obs with mass info)...")
    from algorithms.sac_agent import ReplayBuffer

    class OracleEnv(SpacecraftTakeoverEnvV2):
        """Environment with direct mass observation for oracle upper bound."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            obs_high = np.array(
                [np.inf]*6 + [1.0, np.inf, 1.0, np.inf, 1.0, 1.0],
                dtype=np.float32)
            obs_low = np.array(
                [-np.inf]*6 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32)
            self.observation_space = type(self.observation_space)(low=obs_low, high=obs_high, dtype=np.float32)

        def _get_obs(self):
            base = super()._get_obs()
            dry_norm = (self._dry_mass - 400.0) / 200.0
            fuel_norm = self._fuel_mass / 150.0
            return np.append(base, [dry_norm, fuel_norm]).astype(np.float32)

    oracle_agent = SACAgent(obs_dim=12, action_dim=ACT_DIM, hidden_dim=256,
                            actor_lr=3e-4, critic_lr=3e-4, fixed_alpha=0.2)
    buffer = ReplayBuffer(100000, 12, ACT_DIM)

    env = OracleEnv(max_steps=MAX_STEPS)
    success_count = 0
    for ep in range(200):
        obs, info = env.reset()
        for step in range(MAX_STEPS):
            action = oracle_agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
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

baseline_rate_c3 = baseline_rate
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
