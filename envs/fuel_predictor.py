"""Fuel prediction module for non-cooperative spacecraft.

Estimates remaining fuel from observable dynamics — the target spacecraft's
control torque is reconstructed via rigid-body dynamics, and cumulative fuel
consumption is integrated from torque magnitude, with optional learned correction.

Key insight from ALTITUDE-FIGHT-3D-LQR reference:
  fuel_consumption = k * ||tau|| * dt

For non-cooperative targets, we DON'T have direct fuel telemetry. Instead we:
  1. Estimate target torque from attitude dynamics (already in dynamics.py)
  2. Integrate cumulative torque magnitude → physics-based fuel estimate
  3. Apply learned bias correction for unknown thruster efficiency / ISP
  4. Track prediction uncertainty for robust decision-making
"""

import numpy as np
from collections import deque


class FuelPredictor:
    """Estimates non-cooperative target's remaining fuel from observed torque history.

    Two estimation modes:
      1. Physics-only:  fuel_est = initial_fuel - k * sum(||tau_obs|| * dt)
      2. Learned correction: adds MLP bias correction on top of physics estimate

    Provides both point estimate and uncertainty bounds for robust RL training.
    """

    def __init__(
        self,
        initial_fuel: float = 1.0,
        fuel_k: float = 0.006,
        history_window: int = 100,
        correction_enabled: bool = True,
    ):
        self.initial_fuel = initial_fuel
        self.fuel_k = fuel_k
        self.history_window = history_window

        # Torque history for trend analysis
        self._torque_history: deque[float] = deque(maxlen=history_window)

        # Cumulative integrated values
        self._cumulative_torque_mag: float = 0.0
        self._cumulative_fuel_physics: float = 0.0

        # Prediction state
        self._predicted_remaining: float = initial_fuel
        self._true_remaining: float = initial_fuel
        self._prediction_error: float = 0.0
        self._uncertainty: float = 0.0

        # Learned correction
        self.correction_enabled = correction_enabled
        self._bias_correction: float = 0.0  # learned offset
        self._correction_gain: float = 1.0   # learned multiplicative factor
        self._error_history: deque[float] = deque(maxlen=50)

    def reset(self, initial_fuel: float = None):
        """Reset predictor state for a new episode."""
        if initial_fuel is not None:
            self.initial_fuel = initial_fuel
        self._torque_history.clear()
        self._cumulative_torque_mag = 0.0
        self._cumulative_fuel_physics = 0.0
        self._predicted_remaining = self.initial_fuel
        self._true_remaining = self.initial_fuel
        self._prediction_error = 0.0
        self._uncertainty = 0.0

    def update(
        self,
        observed_target_torque_mag: float,
        dt: float,
        true_fuel_remaining: float = None,
    ) -> dict:
        """Update fuel prediction with latest torque observation.

        Args:
            observed_target_torque_mag: ||tau_target|| estimated from dynamics
            dt: time step
            true_fuel_remaining: ground-truth fuel (for correction learning, if available)

        Returns:
            dict with predicted_remaining, uncertainty, correction info
        """
        self._torque_history.append(observed_target_torque_mag)

        # Physics-based fuel consumption for this step
        fuel_burned_physics = self.fuel_k * observed_target_torque_mag * dt
        self._cumulative_torque_mag += observed_target_torque_mag * dt
        self._cumulative_fuel_physics += fuel_burned_physics

        # Base prediction from physics model
        physics_estimate = self.initial_fuel - self._cumulative_fuel_physics

        # Apply learned correction
        if self.correction_enabled:
            corrected_estimate = physics_estimate - self._bias_correction
            corrected_estimate *= self._correction_gain
        else:
            corrected_estimate = physics_estimate

        self._predicted_remaining = max(0.0, corrected_estimate)

        # Update uncertainty based on torque variability
        if len(self._torque_history) >= 2:
            torque_arr = np.array(self._torque_history)
            torque_std = float(np.std(torque_arr))
            torque_mean = float(np.mean(torque_arr)) + 1e-8
            # Uncertainty grows with torque variability and cumulative time
            self._uncertainty = (
                0.01  # base uncertainty
                + 0.05 * (torque_std / torque_mean)  # relative variability
                + 0.001 * self._cumulative_torque_mag  # drift over time
            )
            self._uncertainty = min(self._uncertainty, 0.5)

        # Update correction if ground truth is available
        if true_fuel_remaining is not None:
            self._true_remaining = true_fuel_remaining
            self._prediction_error = self._predicted_remaining - true_fuel_remaining
            self._error_history.append(self._prediction_error)
            self._update_correction()

        return {
            "predicted_remaining": self._predicted_remaining,
            "physics_estimate": physics_estimate,
            "uncertainty": self._uncertainty,
            "correction_bias": self._bias_correction,
            "correction_gain": self._correction_gain,
            "prediction_error": self._prediction_error,
        }

    def _update_correction(self):
        """Online correction update using exponential moving average of errors."""
        if len(self._error_history) < 5:
            return

        recent_errors = list(self._error_history)[-20:]
        mean_error = float(np.mean(recent_errors))

        # Exponential moving average correction
        alpha = 0.1
        self._bias_correction += alpha * mean_error
        # Clamp correction to reasonable range
        self._bias_correction = np.clip(self._bias_correction, -0.3, 0.3)

        # Adjust gain if error has systematic bias proportional to fuel used
        fuel_used = self.initial_fuel - self._true_remaining
        if fuel_used > 0.05:
            gain_error = mean_error / (fuel_used + 1e-6)
            self._correction_gain += 0.05 * gain_error
            self._correction_gain = np.clip(self._correction_gain, 0.5, 1.5)

    @property
    def predicted(self) -> float:
        return self._predicted_remaining

    @property
    def prediction_normalized(self) -> float:
        """Predicted fuel as fraction [0, 1]."""
        return max(0.0, min(1.0, self._predicted_remaining / max(self.initial_fuel, 1e-6)))

    @property
    def uncertainty_normalized(self) -> float:
        return min(1.0, self._uncertainty)

    def get_features(self) -> np.ndarray:
        """Get prediction features for RL observation.

        Returns 5-dim feature vector:
          [predicted_fuel_normalized, uncertainty, torque_ma, torque_peak, cumulative_torque]
        """
        torque_arr = np.array(self._torque_history) if self._torque_history else np.zeros(1)
        torque_ma = float(np.mean(torque_arr))
        torque_peak = float(np.max(torque_arr)) if len(torque_arr) > 0 else 0.0
        cum_torque_norm = min(1.0, self._cumulative_torque_mag / 50.0)

        return np.array([
            self.prediction_normalized,
            self.uncertainty_normalized,
            min(1.0, torque_ma / 20.0),
            min(1.0, torque_peak / 20.0),
            cum_torque_norm,
        ], dtype=np.float32)
