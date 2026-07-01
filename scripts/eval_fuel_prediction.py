"""Evaluate fuel prediction accuracy on the spacecraft takeover environment.

Runs episodes with different target strategies and measures how well the
FuelPredictor tracks the true remaining fuel of the non-cooperative target.
Generates comparison plots showing predicted vs true fuel over time.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_fuel_prediction(
    episodes: int = 20,
    max_steps: int = 600,
    seed: int = 42,
    save_dir: str = None,
):
    """Evaluate fuel prediction across multiple episodes with varied strategies."""

    env = SpacecraftTakeoverEnvV2(max_steps=max_steps)

    if save_dir is None:
        save_dir = os.path.join("outputs", "fuel_prediction_eval",
                                datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    results = []
    all_pred_errors = []
    all_final_errors = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)

        trajectory = {
            "episode": ep,
            "strategy": info["target_strategy"],
            "steps": [],
            "true_fuel": [],
            "predicted_fuel": [],
            "prediction_error": [],
            "uncertainty": [],
            "torque_mag": [],
        }

        for step in range(max_steps):
            # Use random actions to exercise the predictor across diverse conditions
            action = np.random.uniform(-1.0, 1.0, 3)
            obs, reward, terminated, truncated, info = env.step(action)

            trajectory["steps"].append(step)
            trajectory["true_fuel"].append(info["target_fuel"])
            trajectory["predicted_fuel"].append(info["predicted_fuel"])
            trajectory["prediction_error"].append(info["prediction_error"])
            trajectory["uncertainty"].append(info["prediction_uncertainty"])
            trajectory["torque_mag"].append(info["tau_target_mag"])

            if terminated or truncated:
                break

        final_error = trajectory["prediction_error"][-1]
        mean_abs_error = float(np.mean(np.abs(trajectory["prediction_error"])))

        results.append({
            "episode": ep,
            "strategy": info["target_strategy"],
            "steps": len(trajectory["steps"]),
            "final_true_fuel": trajectory["true_fuel"][-1],
            "final_predicted_fuel": trajectory["predicted_fuel"][-1],
            "final_error": final_error,
            "mean_abs_error": mean_abs_error,
        })
        all_pred_errors.extend(trajectory["prediction_error"])
        all_final_errors.append(final_error)

        print(f"Ep {ep:3d} | {info['target_strategy']:<20s} | "
              f"steps={len(trajectory['steps']):3d} | "
              f"true={trajectory['true_fuel'][-1]:.3f} | "
              f"pred={trajectory['predicted_fuel'][-1]:.3f} | "
              f"error={final_error:+.4f} | MAE={mean_abs_error:.4f}")

        # Save individual episode plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        steps = trajectory["steps"]

        # 1. True vs Predicted Fuel
        axes[0, 0].plot(steps, trajectory["true_fuel"], 'b-', linewidth=2, label='True fuel')
        axes[0, 0].plot(steps, trajectory["predicted_fuel"], 'r--', linewidth=2, label='Predicted fuel')
        axes[0, 0].fill_between(
            steps,
            np.array(trajectory["predicted_fuel"]) - np.array(trajectory["uncertainty"]),
            np.array(trajectory["predicted_fuel"]) + np.array(trajectory["uncertainty"]),
            alpha=0.2, color='red', label='Uncertainty')
        axes[0, 0].set_title(f'Fuel Prediction — {info["target_strategy"]}')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Fuel remaining')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Prediction Error
        axes[0, 1].plot(steps, trajectory["prediction_error"], 'purple', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_title('Prediction Error (predicted - true)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Torque Magnitude
        axes[1, 0].plot(steps, trajectory["torque_mag"], 'orange', linewidth=1.5)
        axes[1, 0].set_title('Target Torque Magnitude')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('||tau||')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Uncertainty over time
        axes[1, 1].plot(steps, trajectory["uncertainty"], 'green', linewidth=1.5)
        axes[1, 1].set_title('Prediction Uncertainty')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Uncertainty')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"ep_{ep:03d}.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # ── Summary statistics ──
    all_mae = [r["mean_abs_error"] for r in results]
    all_final_err = [r["final_error"] for r in results]

    print("\n" + "=" * 60)
    print("PREDICTION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:              {episodes}")
    print(f"Mean MAE:              {np.mean(all_mae):.4f}")
    print(f"Std MAE:               {np.std(all_mae):.4f}")
    print(f"Mean final error:      {np.mean(all_final_err):.4f}")
    print(f"Std final error:       {np.std(all_final_err):.4f}")
    print(f"Max absolute error:    {np.max(np.abs(all_pred_errors)):.4f}")
    print(f"Results saved to:      {save_dir}")

    # Save summary JSON
    summary = {
        "config": {
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
        },
        "statistics": {
            "mean_mae": float(np.mean(all_mae)),
            "std_mae": float(np.std(all_mae)),
            "mean_final_error": float(np.mean(all_final_err)),
            "std_final_error": float(np.std(all_final_err)),
            "max_abs_error": float(np.max(np.abs(all_pred_errors))),
        },
        "results": results,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    env.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fuel prediction accuracy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    evaluate_fuel_prediction(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        save_dir=args.save_dir,
    )
