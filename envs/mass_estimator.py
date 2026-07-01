"""Mass estimator for spacecraft takeover detection.

Estimates target spacecraft mass from observed dynamics to detect
fuel depletion — the trigger condition for initiating takeover.

Physics: τ_total = I·α + ω×(I·ω). The chaser applies known τ_self;
the target applies unknown adversarial τ_target. As the target burns fuel,
its mass decreases → combined inertia I decreases. When fuel is exhausted,
mass stabilizes (dI/dt ≈ 0) → trigger takeover.
"""

import numpy as np


class MassEstimator:
    """Online mass estimator using EMA-smoothed per-axis inertia estimates.

    Estimates the instantaneous moment of inertia from tau_self / alpha
    on each axis, smooths with exponential moving average, and converts
    mean inertia to a mass estimate via inertia_scale. Tracks the time
    derivative of mass over a sliding window to detect stabilisation.

    Parameters
    ----------
    window_size : int
        Number of recent mass samples used for dmass/dt estimation
        and the minimum steps before stability counting begins.
    inertia_scale : float
        Scale factor converting mean inertia to mass (mass = I_mean / scale).
    stability_threshold : float
        |dmass/dt| below this value is considered "stable".
    stability_steps : int
        Consecutive stable steps required to declare fuel depleted.
    """

    def __init__(self, window_size=60, inertia_scale=5.0,
                 stability_threshold=0.001, stability_steps=30,
                 torque_threshold=0.5):
        self.window_size = window_size
        self.inertia_scale = inertia_scale
        self.stability_threshold = stability_threshold
        self.stability_steps = stability_steps
        self.torque_threshold = torque_threshold
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_mass=None):
        """Reset all internal state.

        Parameters
        ----------
        initial_mass : float or None
            If provided, used as the reference for feature normalisation
            and initial inertia estimate.
        """
        self._beta = 0.95                     # EMA decay factor
        self._I_smooth = None                 # (3,) smoothed inertia vector
        self._mass_history = []               # ring buffer of mass estimates
        self._stability_counter = 0
        self._step_count = 0
        self._valid_updates = 0
        self._initial_mass = initial_mass

        # Initialize with reasonable defaults
        default_mass = initial_mass if initial_mass is not None else 100.0
        self._mass_est = default_mass
        # I_diag = mass * inertia_scale * [1.0, 1.0, 1.0]
        self._I_est = np.full(3, default_mass * self.inertia_scale, dtype=np.float64)
        self._I_smooth = self._I_est.copy()
        self._dmass_dt = 0.0

        # Clear windowed LS buffers
        self._alpha_buf = [[] for _ in range(3)]
        self._tau_buf = [[] for _ in range(3)]
        # Clear dmass/dt smoother
        self._dmass_dt_smooth = 0.0

    def update(self, omega, alpha, tau_self, dt):
        """Ingest one dynamics step and update all estimates.

        Parameters
        ----------
        omega : array-like (3,)
            Angular velocity in rad/s (MuJoCo ``qvel``).
        alpha : array-like (3,)
            Angular acceleration in rad/s² (finite-difference of omega).
        tau_self : array-like (3,)
            Self-applied torque in N·m (known chaser actuation).
        dt : float
            Time step in seconds.

        Returns
        -------
        dict
            Keys: ``mass_est``, ``I_est``, ``dmass_dt``, ``is_stable``,
            ``step_count``.
        """
        omega = np.asarray(omega, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        tau_self = np.asarray(tau_self, dtype=np.float64)

        self._step_count += 1

        # ---- HIGH-SNR MOMENT DETECTION ----
        # Only update I when ||tau_self|| AND ||alpha|| are large enough.
        tau_norm = float(np.linalg.norm(tau_self))
        alpha_norm = float(np.linalg.norm(alpha))

        if tau_norm >= self.torque_threshold and alpha_norm > 0.005:
            # ---- compute gyroscopic term using current I estimate ----
            I_omega = self._I_smooth * omega
            gyro = np.cross(omega, I_omega)

            # ---- Net known torque (subtract gyro to isolate inertial term) ----
            tau_net = tau_self - gyro

            # ---- Accumulate (alpha, tau_net) pairs for windowed LS ----

            for i in range(3):
                self._alpha_buf[i].append(float(alpha[i]))
                self._tau_buf[i].append(float(tau_net[i]))
                if len(self._alpha_buf[i]) > self.window_size:
                    self._alpha_buf[i].pop(0)
                    self._tau_buf[i].pop(0)

            # ---- Windowed Least Squares: I[i] = slope of tau vs alpha ----
            for i in range(3):
                if len(self._alpha_buf[i]) >= 10:
                    a_arr = np.array(self._alpha_buf[i])
                    t_arr = np.array(self._tau_buf[i])
                    # I = tau/alpha, least squares: min Σ(tau - I*alpha)²
                    # Solution: I = Σ(α·τ) / Σ(α²)
                    num = float(np.dot(a_arr, t_arr))
                    den = float(np.dot(a_arr, a_arr))
                    if den > 1e-8:
                        I_ls = num / den
                        # Clamp to physically plausible
                        I_ls = float(np.clip(I_ls,
                            self._I_smooth[i] * 0.5,
                            self._I_smooth[i] * 1.5))
                        # EMA smooth
                        self._I_smooth[i] = (
                            0.9 * self._I_smooth[i] + 0.1 * I_ls
                        )

            self._valid_updates += 1

        # ---- After first high-SNR update, continue tracking ----
        # Mass from mean inertia
        self._I_est = self._I_smooth.copy()
        i_mean = float(np.mean(self._I_est))
        self._mass_est = i_mean / self.inertia_scale

        self._mass_history.append(self._mass_est)
        if len(self._mass_history) > self.window_size:
            self._mass_history = self._mass_history[-self.window_size:]

        # ---- dmass / dt over longer window (for better noise rejection) ----
        dmass_window = self.window_size * 2
        if len(self._mass_history) >= dmass_window:
            mass_window_ago = self._mass_history[-dmass_window]
            window_dt = dmass_window * dt
            if window_dt > 1e-12:
                raw_dmass_dt = (self._mass_est - mass_window_ago) / window_dt
                # EMA smooth dmass/dt to reject LS scatter
                if not hasattr(self, '_dmass_dt_smooth'):
                    self._dmass_dt_smooth = raw_dmass_dt
                self._dmass_dt_smooth = (
                    0.8 * self._dmass_dt_smooth + 0.2 * raw_dmass_dt
                )
                self._dmass_dt = self._dmass_dt_smooth
        elif len(self._mass_history) >= self.window_size:
            mass_window_ago = self._mass_history[0]
            window_dt = self.window_size * dt
            if window_dt > 1e-12:
                self._dmass_dt = (self._mass_est - mass_window_ago) / window_dt

        # ---- stability detection (dual criterion) ----
        # Criterion 1: dmass/dt magnitude below threshold
        is_stable_dmass = abs(self._dmass_dt) < self.stability_threshold
        
        # Criterion 2: mass variance within window is low (converged)
        mass_variance = 0.0
        if len(self._mass_history) >= self.window_size // 2:
            mass_variance = float(np.var(self._mass_history))
        is_stable_variance = mass_variance < 1.0  # mass stable within ~1 kg²

        is_stable = is_stable_dmass and is_stable_variance

        # Only start stability counting after we have enough valid updates
        # (i.e., we have actually been tracking mass changes)
        if self._step_count >= self.window_size and self._valid_updates >= self.window_size // 2:
            if is_stable:
                self._stability_counter += 1
            else:
                self._stability_counter = 0
        else:
            # Ignore initial transient — not enough history yet
            self._stability_counter = 0

        return {
            'mass_est': self._mass_est,
            'I_est': self._I_est.copy(),
            'dmass_dt': self._dmass_dt,
            'is_stable': is_stable,
            'is_stable_dmass': is_stable_dmass,
            'is_stable_variance': is_stable_variance,
            'mass_variance': mass_variance,
            'step_count': self._step_count,
        }

    def is_fuel_depleted(self):
        """Return True when mass has stabilised for enough consecutive steps."""
        return self._stability_counter >= self.stability_steps

    def get_features(self):
        """Normalised feature vector for downstream policies.

        Returns
        -------
        np.ndarray, shape (5,)
            ``[mass_norm, dmass_dt_norm, i_mean_norm, confidence,
            stability_norm]``
        """
        mass_ref = (
            self._initial_mass
            if self._initial_mass is not None
            else max(self._mass_est, 1e-6)
        )
        mass_norm = self._mass_est / max(mass_ref, 1e-6)

        # Clip dmass/dt to [-1, 1] relative to 10× threshold
        dmass_dt_norm = float(np.clip(
            self._dmass_dt / max(self.stability_threshold * 10.0, 1e-8),
            -1.0, 1.0,
        ))

        i_mean = float(np.mean(self._I_est))
        i_mean_norm = i_mean / max(self.inertia_scale * mass_ref, 1e-6)

        confidence = min(self._step_count / max(self.window_size, 1), 1.0)

        stability_norm = min(
            self._stability_counter / max(self.stability_steps, 1), 1.0
        )

        return np.array(
            [mass_norm, dmass_dt_norm, i_mean_norm, confidence, stability_norm],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass_est(self):
        """Current mass estimate (scalar)."""
        return self._mass_est

    @property
    def I_est(self):
        """Current smoothed inertia vector (3,)."""
        return self._I_est.copy()

    @property
    def dmass_dt(self):
        """Current mass time-derivative estimate."""
        return self._dmass_dt

    @property
    def stability_counter(self):
        """Consecutive steps with |dmass/dt| below threshold."""
        return self._stability_counter

    @property
    def valid_updates(self):
        """Number of high-SNR update steps executed."""
        return self._valid_updates


# ---------------------------------------------------------------------------
# __main__: smoke test — linearly decreasing mass → plateau → detection
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(42)

    TOTAL_STEPS = 500
    DT = 0.02          # 20 ms → 50 Hz
    DEPLETE_STEP = 300

    # True mass: linear ramp then flat
    M0, M1 = 100.0, 50.0
    true_mass = np.where(
        np.arange(TOTAL_STEPS) < DEPLETE_STEP,
        M0 + (M1 - M0) * (np.arange(TOTAL_STEPS) / DEPLETE_STEP),
        M1,
    )

    # Anisotropic inertia: I = mass * inertia_scale * diag(1.0, 1.2, 0.8)
    SCALE = 5.0
    true_I = np.stack([
        true_mass * SCALE * 1.0,
        true_mass * SCALE * 1.2,
        true_mass * SCALE * 0.8,
    ], axis=1)  # (steps, 3)

    est = MassEstimator(
        window_size=60,
        inertia_scale=SCALE,
        stability_threshold=0.001,
        stability_steps=30,
    )
    est.reset(initial_mass=M0)

    omega = np.array([0.1, -0.05, 0.03])  # small constant rotation

    mass_ests, dmass_dts, depleted_flags = [], [], []

    for t in range(TOTAL_STEPS):
        I_diag = true_I[t]                     # (3,)
        I_mat = np.diag(I_diag)                # (3,3)

        # Fictional chaser torque (oscillatory)
        tau_self = np.array([
            0.5 * np.sin(2 * np.pi * 0.10 * t * DT),
            0.3 * np.cos(2 * np.pi * 0.15 * t * DT),
            -0.2 * np.sin(2 * np.pi * 0.08 * t * DT + 0.5),
        ])

        # Compute true α from  τ = I·α + ω×(I·ω)
        gyro = np.cross(omega, I_mat @ omega)
        tau_net = tau_self - gyro
        alpha_true = np.linalg.solve(I_mat, tau_net)

        # Measured α with noise (simulating finite-difference errors)
        alpha_meas = alpha_true + np.random.normal(0, 0.02, 3)

        est.update(omega, alpha_meas, tau_self, DT)
        mass_ests.append(est.mass_est)
        dmass_dts.append(est.dmass_dt)
        depleted_flags.append(est.is_fuel_depleted())

    first_depleted = next(
        (i for i, d in enumerate(depleted_flags) if d), None
    )

    print(f"Fuel depletion detected at step {first_depleted}"
          if first_depleted else "Fuel depletion NOT detected")
    print(f"True depletion at step          {DEPLETE_STEP}")
    print(f"Final mass estimate:            {est.mass_est:.3f}  "
          f"(true: {M1:.3f})")
    print(f"Final inertia estimate:         {est.I_est}")
    print(f"Features:                       {est.get_features()}")

    # ---- plot ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(true_mass, 'k--', alpha=0.5, label='True mass')
    ax.plot(mass_ests, 'b-', alpha=0.7, label='Estimated mass')
    ax.axvline(DEPLETE_STEP, color='r', ls=':', alpha=0.5,
               label='True depletion')
    if first_depleted:
        ax.axvline(first_depleted, color='g', ls='--', alpha=0.5,
                   label='Detected depletion')
    ax.set_ylabel('Mass (kg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mass Estimator — Linear Decrease → Plateau')

    ax = axes[1]
    ax.plot(dmass_dts, 'r-', alpha=0.7, label='dmass/dt')
    ax.axhline(0, color='k', ls=':', alpha=0.3)
    ax.axhline(+est.stability_threshold, color='orange', ls='--', alpha=0.5,
               label=f'±{est.stability_threshold}')
    ax.axhline(-est.stability_threshold, color='orange', ls='--', alpha=0.5)
    ax.set_ylabel('dmass / dt')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.fill_between(range(TOTAL_STEPS), 0,
                    [1.0 if f else 0.0 for f in depleted_flags],
                    alpha=0.3, color='green', label='Fuel depleted')
    ax.set_ylabel('Depletion')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'mass_estimator_test.png'
    plt.savefig(out, dpi=100)
    print(f"Plot saved to {out}")
    plt.show()
