# envs/spacecraft_env_v5.py
"""Gymnasium env — V5: claw mechanism, efficiency reward, TD3-oriented.

Key changes from V4:
  - Claw MJCF model (claw_body.xml)
  - Efficiency reward: fuel_burned / self_burn
  - W_SELF_BURN ×5, self_burn_rate ×2
  - MAX_ATT_ERR = pi/2 (tighter constraint)
  - Mixed target strategies (LQR + SMC + PID)
  - self_fuel is the real time limit, max_steps is safety net
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from envs.dynamics import quat_to_mrp, mrp_error, compute_gravity_gradient_torque
from algorithms.target_controllers import (
    make_target_strategies_strong, make_target_strategies,
    TargetLQRController,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "claw_body.xml")


class SpacecraftTakeoverEnvV5(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ── reward weights (V5: efficiency-oriented) ─────────────────────────
    W_FUEL_BURN  = 10.0    # opponent fuel burned (kg)
    W_SELF_BURN  = 3.0     # self fuel burned penalty
    W_STEP       = 0.01    # per-step time cost
    W_MASS       = 0.5     # mass-estimation auxiliary signal
    W_EFF        = 2.0     # efficiency bonus
    W_ATT        = 0.3     # attitude penalty (continuous, to prevent tumbling)
    R_SUCCESS    = 200.0   # opponent fuel fully depleted
    R_DETECT     = 100.0   # dry_mass detected → phase switch
    R_FAIL       = -100.0  # self fuel gone or tumbled

    # ── physical parameters ──────────────────────────────────────────────
    MAX_TORQUE     = 5.0
    FUEL_K_MASS    = 3.0     # kg fuel per (N·m·s) — faster target depletion
    SELF_BURN_RATE = 0.012   # self fuel burn rate (×2 from V4)
    INITIAL_FUEL   = 1.0     # chaser's own fuel (normalized)
    MAX_ATT_ERR    = 2.0     # ~115° — still constraining, but more forgiving

    CTL_DT   = 1.0 / 60.0
    SUBSTEPS = 30

    # ── mass randomization ───────────────────────────────────────────────
    DRY_MASS_MIN = 400.0
    DRY_MASS_MAX = 600.0
    DRY_MASS_EPISODES = 100
    FUEL_MASS_MIN = 100.0
    FUEL_MASS_MAX = 150.0

    # ── mass estimation / phase switch ───────────────────────────────────
    MASS_HISTORY_WINDOW = 80
    MASS_STABLE_STEPS   = 10
    MASS_STABLE_EPS     = 3.0
    TAU_RESPONSE_FLOOR  = 0.1

    def __init__(self, render_mode=None, max_steps=600):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = self.CTL_DT / self.SUBSTEPS

        # ── save original MJCF target body params ─────────────────────────
        self._target_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sat")
        self._orig_target_mass = float(self.model.body_mass[self._target_bid])
        self._orig_target_inertia = self.model.body_inertia[self._target_bid].copy()
        self._orig_inertia_ratio = self._orig_target_inertia / self._orig_target_mass

        # ── observation: 10 dim ───────────────────────────────────────────
        obs_high = np.array([np.inf]*6 + [1.0, np.inf, 1.0, np.inf], dtype=np.float32)
        obs_low  = np.array([-np.inf]*6 + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # ── target strategies: LQR + SMC + PID mixed ─────────────────────
        strong = make_target_strategies_strong()
        base = make_target_strategies()
        self.target_strategies = strong + base

        # internal state
        self._step_count    = 0
        self._self_fuel     = 1.0
        self._target_fuel   = 1.0
        self._prev_omega    = np.zeros(3)
        self._alpha         = np.zeros(3)
        self._last_tau_self_mag = 0.0
        self._mass_response = 0.0

        # mass tracking
        self._dry_mass       = 500.0
        self._fuel_mass_init = 100.0
        self._fuel_mass      = 100.0
        self._total_mass     = 600.0
        self._episode_count  = 0
        self._cumulative_target_torque = 0.0
        self._mass_fuel_k    = 3.0

        self._est_mass       = 600.0
        self._est_mass_ema   = 600.0
        self._mass_history   = []
        self._mass_stable_ctr = 0
        self._phase_switched  = False
        self._bonus_given     = False

    # ── Gym API ──────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random if self.np_random is not None else np.random

        self._episode_count += 1
        if self._episode_count % self.DRY_MASS_EPISODES == 1 or self._episode_count == 1:
            self._dry_mass = float(rng.uniform(self.DRY_MASS_MIN, self.DRY_MASS_MAX))

        self._fuel_mass_init = float(rng.uniform(self.FUEL_MASS_MIN, self.FUEL_MASS_MAX))
        self._fuel_mass = self._fuel_mass_init
        self._total_mass = self._dry_mass + self._fuel_mass

        bid = self._target_bid
        self.model.body_mass[bid] = self._total_mass
        self.model.body_inertia[bid] = self._orig_inertia_ratio * self._total_mass

        mujoco.mj_resetData(self.model, self.data)
        sigma_init = rng.uniform(-0.15, 0.15, 3)
        self.data.qpos[3:7] = self._mrp2quat(sigma_init)
        self.data.qvel[0:3] = rng.uniform(-0.1, 0.1, 3)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._self_fuel  = self.INITIAL_FUEL
        self._target_fuel = 1.0
        self._prev_omega   = self.data.qvel[0:3].copy()
        self._alpha        = np.zeros(3)
        self._last_tau_self_mag = 0.0
        self._mass_response = 0.0

        self._cumulative_target_torque = 0.0
        self._mass_initial_guess = self._total_mass
        self._est_mass = self._total_mass
        self._est_mass_ema = self._total_mass
        self._mass_history.clear()
        self._mass_stable_ctr = 0
        self._phase_switched = False
        self._bonus_given = False

        idx = rng.integers(0, len(self.target_strategies))
        self._current_strategy = self.target_strategies[idx]
        if hasattr(self._current_strategy, 'reset'):
            self._current_strategy.reset()
        if hasattr(self._current_strategy, 'set_inertia'):
            self._current_strategy.set_inertia(
                np.diag(self.model.body_inertia[bid]))

        return self._get_obs(), {
            "self_fuel": self._self_fuel,
            "dry_mass": self._dry_mass,
            "fuel_mass": self._fuel_mass,
            "total_mass": self._total_mass,
            "target_strategy": type(self._current_strategy).__name__,
        }

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64)
        action_clipped = np.clip(action, -1.0, 1.0)
        tau_self = action_clipped * self.MAX_TORQUE
        tau_self_mag = float(np.linalg.norm(tau_self))
        self._last_tau_self_mag = tau_self_mag

        # ── target response ───────────────────────────────────────────────
        sigma_cur  = self._get_mrp()
        omega      = self.data.qvel[0:3].copy()
        sigma_err  = mrp_error(sigma_cur, np.zeros(3))
        tau_target = self._current_strategy.compute(sigma_err, omega, dt=self.CTL_DT)

        # ── MuJoCo simulation ─────────────────────────────────────────────
        tau_total = tau_self + tau_target
        bid_svc = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        for _ in range(self.SUBSTEPS):
            self.data.xfrc_applied[bid_svc, 0:3] = tau_total
            self.data.ctrl[0:3] = tau_total
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)

        omega_new = self.data.qvel[0:3].copy()

        # ── self fuel burn (V5: doubled rate) ─────────────────────────────
        self_burn = float(self.CTL_DT * tau_self_mag * self.SELF_BURN_RATE)
        self._self_fuel -= self_burn

        # ── target fuel burn (V5: k=3.0 for faster depletion) ────────────
        tau_mag = float(np.linalg.norm(tau_target))
        fuel_burned_kg = self.FUEL_K_MASS * tau_mag * self.CTL_DT
        self._fuel_mass = max(0.0, self._fuel_mass - fuel_burned_kg)
        self._total_mass = self._dry_mass + self._fuel_mass
        self._target_fuel = self._fuel_mass / max(self._fuel_mass_init, 1e-6)

        # ── update MJCF ───────────────────────────────────────────────────
        bid = self._target_bid
        self.model.body_mass[bid] = self._total_mass
        self.model.body_inertia[bid] = self._orig_inertia_ratio * self._total_mass
        mujoco.mj_forward(self.model, self.data)

        if self._fuel_mass > 0:
            self._cumulative_target_torque += tau_mag * self.CTL_DT

        # ── angular acceleration ──────────────────────────────────────────
        self._alpha = (omega_new - self._prev_omega) / self.CTL_DT
        self._prev_omega = omega_new.copy()

        if tau_self_mag > self.TAU_RESPONSE_FLOOR:
            raw_response = float(np.linalg.norm(self._alpha) / (tau_self_mag + 0.01))
            self._mass_response = min(raw_response, 0.2)
        else:
            self._mass_response = 0.0

        # ── mass estimation ───────────────────────────────────────────────
        cum_torque = self._cumulative_target_torque
        fuel_burned_est = cum_torque * self._mass_fuel_k
        self._est_mass = self._mass_initial_guess - fuel_burned_est
        self._est_mass_ema = 0.9 * self._est_mass_ema + 0.1 * self._est_mass

        # ── phase switch ──────────────────────────────────────────────────
        self._mass_history.append(self._est_mass)
        if len(self._mass_history) > self.MASS_HISTORY_WINDOW:
            self._mass_history.pop(0)

        if len(self._mass_history) >= self.MASS_HISTORY_WINDOW // 2:
            half = len(self._mass_history) // 2
            recent = np.mean(self._mass_history[-half:])
            older  = np.mean(self._mass_history[:half])
            mass_change = abs(recent - older)
            if mass_change < self.MASS_STABLE_EPS:
                self._mass_stable_ctr += 1
            else:
                self._mass_stable_ctr = max(0, self._mass_stable_ctr - 2)

        if not self._phase_switched and self._mass_stable_ctr >= self.MASS_STABLE_STEPS:
            self._phase_switched = True

        # ── reward (V5: efficiency-oriented) ──────────────────────────────
        SCALE = 0.01
        reward = SCALE * (self.W_FUEL_BURN * fuel_burned_kg
                          - self.W_SELF_BURN * self_burn
                          - self.W_STEP)

        # Efficiency bonus: bounded ratio ∈ [0, 1]
        efficiency = fuel_burned_kg / max(self_burn + fuel_burned_kg, 1e-8)
        reward += SCALE * self.W_EFF * efficiency

        # Attitude penalty: continuous signal to prevent tumbling
        reward -= SCALE * self.W_ATT * att_err

        # Mass estimation auxiliary
        mass_error = (self._est_mass - self._dry_mass)
        reward -= SCALE * self.W_MASS * (mass_error / max(self._dry_mass, 1.0)) ** 2

        # Phase switch bonus
        if self._phase_switched and not self._bonus_given:
            reward += SCALE * self.R_DETECT
            self._bonus_given = True

        # ── terminal ──────────────────────────────────────────────────────
        terminated = False
        truncated  = self._step_count >= self.max_steps
        att_err    = float(np.linalg.norm(sigma_err))

        if self._fuel_mass <= 0.0:
            if not self._phase_switched:
                self._phase_switched = True
            reward += SCALE * self.R_SUCCESS
            self._fuel_mass = 0.0
            self._total_mass = self._dry_mass
        elif self._self_fuel <= 0.0:
            terminated = True
            reward += SCALE * self.R_FAIL
        elif att_err > self.MAX_ATT_ERR:
            terminated = True
            reward += SCALE * self.R_FAIL

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {
            "self_fuel": self._self_fuel,
            "target_fuel": self._target_fuel,
            "tau_target_mag": tau_mag,
            "attitude_error": att_err,
            "reward": reward,
            "target_strategy": type(self._current_strategy).__name__,
            "self_burn": self_burn,
            "target_burn_kg": fuel_burned_kg,
            "dry_mass": self._dry_mass,
            "fuel_mass": self._fuel_mass,
            "total_mass": self._total_mass,
            "est_mass": self._est_mass,
            "mass_error": mass_error,
            "mass_response": self._mass_response,
            "phase_switched": self._phase_switched,
            "mass_stable_ctr": self._mass_stable_ctr,
            "efficiency": efficiency,
        }

    # ── observation ──────────────────────────────────────────────────────
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
        return obs

    def _get_mrp(self) -> np.ndarray:
        return quat_to_mrp(self.data.qpos[3:7].copy())

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 1920, 1080)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            renderer.close()
            return pixels
        return None

    def close(self):
        pass

    @staticmethod
    def _mrp2quat(sigma):
        s2 = np.dot(sigma, sigma)
        denom = 1.0 + s2
        return np.array([(1.0 - s2) / denom,
                         2.0 * sigma[0] / denom,
                         2.0 * sigma[1] / denom,
                         2.0 * sigma[2] / denom])
