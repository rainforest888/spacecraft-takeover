"""Response-based mass change detector for non-cooperative spacecraft.

Instead of estimating absolute inertia (impossible without knowing τ_target),
we track the RATIO of angular acceleration to self-applied torque:

    response_ratio = ||alpha|| / ||tau_self||

This is proportional to 1/I_eff (effective combined inertia). As target burns
fuel, mass decreases → I decreases → response_ratio increases. When fuel is
exhausted → mass stabilizes → response_ratio stabilizes → trigger takeover.

This approach is robust to unknown τ_target because:
  - At moments when ||tau_self|| >> ||tau_target||, response_ratio ≈ 1/I
  - Even with τ_target noise, the TREND of response_ratio reflects mass changes
  - We only track stability of the trend, not absolute values
"""

import numpy as np
from collections import deque


class MassChangeDetector:
    """Detects mass stabilisation from response ratio: ||alpha|| / ||tau_self||.

    When this ratio stops changing significantly, the target's mass has
    plateaued → fuel depleted → takeover should begin.
    """

    def __init__(self, window_size=120, stability_threshold=0.05,
                 stability_steps=30, tau_threshold=2.0):
        """
        Args:
            window_size: samples in sliding window for trend estimation
            stability_threshold: max |d(ratio)/dt| for "stable"
            stability_steps: consecutive stable steps to declare depleted
            tau_threshold: min ||tau_self|| to accept a measurement (high SNR)
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.stability_steps = stability_steps
        self.tau_threshold = tau_threshold
        self.reset()

    def reset(self):
        self._ratio_history = deque(maxlen=self.window_size)
        self._ratio_ema = None       # EMA-smoothed ratio
        self._ratio_dot = 0.0        # d(ratio)/dt
        self._ratio_dot_ema = 0.0    # smoothed derivative
        self._stability_counter = 0
        self._step_count = 0
        self._valid_steps = 0
        self._ratio = 0.0

    def update(self, alpha: np.ndarray, tau_self: np.ndarray,
               dt: float) -> dict:
        """Update detector with latest dynamics step.

        Args:
            alpha: angular acceleration (3,)
            tau_self: self-applied torque (3,)
            dt: time step

        Returns:
            dict with ratio, ratio_dot, is_stable, valid_steps
        """
        self._step_count += 1
        alpha_norm = float(np.linalg.norm(alpha))
        tau_norm = float(np.linalg.norm(tau_self))

        # Only use high-SNR measurements
        if tau_norm >= self.tau_threshold and alpha_norm > 1e-8:
            ratio = tau_norm / alpha_norm  # ~ I_eff (effective inertia proxy)

            # EMA smooth the ratio
            if self._ratio_ema is None:
                self._ratio_ema = ratio
            else:
                self._ratio_ema = 0.95 * self._ratio_ema + 0.05 * ratio

            self._ratio_history.append(self._ratio_ema)
            self._valid_steps += 1
            self._ratio = self._ratio_ema

        # Compute d(ratio)/dt over window
        if len(self._ratio_history) >= self.window_size // 2:
            ratio_arr = np.array(self._ratio_history)
            # Linear fit slope = d(ratio)/dt
            n = len(ratio_arr)
            t = np.arange(n) * dt
            # Least squares slope
            t_mean = np.mean(t)
            r_mean = np.mean(ratio_arr)
            num = np.sum((t - t_mean) * (ratio_arr - r_mean))
            den = np.sum((t - t_mean) ** 2)
            if den > 1e-12:
                raw_dot = num / den
                # EMA smooth derivative
                self._ratio_dot_ema = (
                    0.9 * self._ratio_dot_ema + 0.1 * raw_dot
                )
                self._ratio_dot = self._ratio_dot_ema

        # Stability detection
        is_stable = abs(self._ratio_dot) < self.stability_threshold

        if self._valid_steps >= self.window_size // 2:
            if is_stable:
                self._stability_counter += 1
            else:
                self._stability_counter = max(0, self._stability_counter - 2)
        else:
            self._stability_counter = 0

        return {
            "ratio": self._ratio,
            "ratio_dot": self._ratio_dot,
            "is_stable": is_stable,
            "stability_counter": self._stability_counter,
            "valid_steps": self._valid_steps,
        }

    def is_depleted(self) -> bool:
        """True when response ratio has stabilised (mass stopped changing)."""
        return self._stability_counter >= self.stability_steps

    def get_features(self) -> np.ndarray:
        """Normalised features for RL observation: [ratio_norm, ratio_dot_norm, stability_norm]."""
        ratio_norm = min(1.0, self._ratio / 1000.0) if self._ratio > 0 else 0.5
        ratio_dot_norm = float(np.clip(
            self._ratio_dot / max(self.stability_threshold * 10.0, 1e-8), -1.0, 1.0))
        stability_norm = min(self._stability_counter / max(self.stability_steps, 1), 1.0)
        return np.array([ratio_norm, ratio_dot_norm, stability_norm], dtype=np.float64)

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def ratio_dot(self) -> float:
        return self._ratio_dot

    @property
    def stability_counter(self) -> int:
        return self._stability_counter
