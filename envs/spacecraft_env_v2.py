"""Gymnasium env — dynamic target mass with unknown dry_mass estimation.

Core concept:
  - dry_mass: baseline mass (unknown to agent, changes slowly across episodes)
  - fuel_mass: randomly sampled propellant mass (unknown to agent)
  - total_mass = dry_mass + fuel_mass  →  set on MJCF target_sat body

  Each step: fuel burns → MJCF body_mass & body_inertia decrease in real time.
  Agent observes dynamics → estimates current mass.
  As fuel depletes, estimated mass converges to dry_mass → phase switch → attitude hold.

  Mass loss penalty: (estimated_mass - dry_mass)^2, driving the network to learn
  mass estimation from dynamics alone.

Observation: [sigma(3), ω(3), self_fuel(1), est_mass_norm(1), mass_change_rate(1),
              tau_ma(1), dmass_detection(1), t_norm(1), att_err(1)] = 13 dims
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from envs.dynamics import (
    quat_to_mrp, mrp_error,
    compute_gravity_gradient_torque,
)
from algorithms.target_controllers import make_target_strategies_strong

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "combo_body.xml")


class SpacecraftTakeoverEnvV2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ── reward weights ─────────────────────────────────────────────────
    W_FUEL      = 500.0    # excess target burn
    W_SELF      = 15.0     # self fuel penalty
    W_LYAP      = 10.0     # attitude error
    W_SMOOTH    = 3.0      # action smoothness
    W_MASS      = 200.0    # mass estimation accuracy penalty
    R_SUCCESS   = 200.0    # target fuel fully depleted
    R_FAIL      = -50.0    # self fuel gone or tumbled
    R_DETECT    = 300.0    # dry_mass detected → phase switch

    # ── physical parameters ────────────────────────────────────────────
    MAX_TORQUE   = 12.0
    FUEL_K_MASS  = 2.0     # kg fuel burned per (N·m · s)
    INITIAL_FUEL = 1.0     # chaser's own fuel (normalized)
    MAX_ATT_ERR  = np.pi

    CTL_DT   = 1.0 / 60.0
    SUBSTEPS = 30
    BASELINE_BURN = 0.0003

    # ── mass randomization ranges ──────────────────────────────────────
    # dry_mass changes every DRY_MASS_EPISODES (slowly-varying unknown)
    DRY_MASS_MIN = 400.0   # kg
    DRY_MASS_MAX = 600.0   # kg
    DRY_MASS_EPISODES = 100
    FUEL_MASS_MIN = 50.0   # kg
    FUEL_MASS_MAX = 200.0  # kg

    # ── mass estimation / phase switch ─────────────────────────────────
    MASS_HISTORY_WINDOW = 80
    MASS_STABLE_STEPS   = 10   # consecutive stable steps → detected
    MASS_STABLE_EPS     = 3.0   # kg: mass change < this = stable

    def __init__(self, render_mode=None, max_steps=600):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = self.CTL_DT / self.SUBSTEPS

        # ── save original MJCF target body params ───────────────────────
        self._target_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sat")
        self._orig_target_mass = float(self.model.body_mass[self._target_bid])
        self._orig_target_inertia = self.model.body_inertia[self._target_bid].copy()
        self._orig_inertia_ratio = self._orig_target_inertia / self._orig_target_mass

        # ── observation: 13 dim ─────────────────────────────────────────
        obs_high = np.array(
            [np.inf]*6 + [1.0, 1.0, 1.0, np.inf, 1.0, 1.0, np.inf],
            dtype=np.float32)
        obs_low  = np.array(
            [-np.inf]*6 + [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.target_strategies = make_target_strategies_strong()
        self._current_strategy = None

        # internal state
        self._step_count    = 0
        self._self_fuel     = 1.0
        self._target_fuel   = 1.0
        self._tau_history   = []
        self._tau_ma        = 0.0
        self._prev_tau_mag  = 0.0
        self._prev_action   = np.zeros(3)
        self._prev_omega    = np.zeros(3)
        self._alpha         = np.zeros(3)

        # ── mass tracking ───────────────────────────────────────────────
        self._dry_mass       = 500.0
        self._fuel_mass_init = 100.0
        self._fuel_mass      = 100.0
        self._total_mass     = 600.0
        self._episode_count  = 0
        self._cumulative_target_torque = 0.0  # Σ ||τ_target|| * dt
        self._mass_fuel_k    = 2.0  # kg fuel per (N·m·s) — matches FUEL_K_MASS

        # Mass estimation: simple EMA of dynamic response ratio
        self._est_mass       = 600.0      # agent's mass estimate
        self._est_mass_ema   = 600.0
        self._mass_history   = []         # sliding window of estimates
        self._mass_stable_ctr = 0
        self._phase_switched  = False
        self._bonus_given     = False

    # ── Gym API ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random if self.np_random is not None else np.random

        # ── randomize dry_mass (changes slowly across episodes) ─────────
        self._episode_count += 1
        if self._episode_count % self.DRY_MASS_EPISODES == 1 or self._episode_count == 1:
            self._dry_mass = float(rng.uniform(self.DRY_MASS_MIN, self.DRY_MASS_MAX))

        # ── randomize fuel_mass ─────────────────────────────────────────
        self._fuel_mass_init = float(rng.uniform(self.FUEL_MASS_MIN, self.FUEL_MASS_MAX))
        self._fuel_mass = self._fuel_mass_init
        self._total_mass = self._dry_mass + self._fuel_mass

        # ── set MJCF target body mass & inertia ─────────────────────────
        bid = self._target_bid
        self.model.body_mass[bid] = self._total_mass
        self.model.body_inertia[bid] = self._orig_inertia_ratio * self._total_mass

        # ── reset MuJoCo state ──────────────────────────────────────────
        mujoco.mj_resetData(self.model, self.data)
        sigma_init = rng.uniform(-0.15, 0.15, 3)
        self.data.qpos[3:7] = self._mrp2quat(sigma_init)
        self.data.qvel[0:3] = rng.uniform(-0.1, 0.1, 3)
        mujoco.mj_forward(self.model, self.data)

        # ── reset episode state ─────────────────────────────────────────
        self._step_count = 0
        self._self_fuel  = self.INITIAL_FUEL
        self._target_fuel = 1.0
        self._tau_history.clear()
        self._tau_ma       = 0.0
        self._prev_tau_mag = 0.0
        self._prev_action  = np.zeros(3)
        self._prev_omega   = self.data.qvel[0:3].copy()
        self._alpha        = np.zeros(3)

        # Mass estimation init
        self._cumulative_target_torque = 0.0
        self._mass_initial_guess = self._total_mass
        self._est_mass = self._total_mass
        self._est_mass_ema = self._total_mass
        self._mass_history.clear()
        self._mass_stable_ctr = 0
        self._phase_switched = False
        self._bonus_given = False

        # ── pick target strategy ────────────────────────────────────────
        idx = rng.integers(0, len(self.target_strategies))
        self._current_strategy = self.target_strategies[idx]
        if hasattr(self._current_strategy, 'reset'):
            self._current_strategy.reset()
        if hasattr(self._current_strategy, 'set_inertia'):
            self._current_strategy.set_inertia(
                np.diag(self.model.body_inertia[bid]))

        return self._get_obs(), {
            "self_fuel": self._self_fuel,
            "dry_mass": self._dry_mass,           # unknown to agent!
            "fuel_mass": self._fuel_mass,          # unknown to agent!
            "total_mass": self._total_mass,        # unknown to agent!
            "target_strategy": type(self._current_strategy).__name__,
        }

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64)
        action_clipped = np.clip(action, -1.0, 1.0)
        tau_self = action_clipped * self.MAX_TORQUE

        # ── target response ────────────────────────────────────────────
        sigma_cur  = self._get_mrp()
        omega      = self.data.qvel[0:3].copy()
        sigma_err  = mrp_error(sigma_cur, np.zeros(3))
        tau_target = self._current_strategy.compute(sigma_err, omega, dt=self.CTL_DT)

        # ── MuJoCo simulation ──────────────────────────────────────────
        tau_total = tau_self + tau_target
        bid_svc = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        for _ in range(self.SUBSTEPS):
            self.data.xfrc_applied[bid_svc, 0:3] = tau_total
            self.data.ctrl[0:3] = tau_total
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)

        omega_new = self.data.qvel[0:3].copy()

        # ── chaser fuel ─────────────────────────────────────────────────
        self_burn = float(self.CTL_DT * np.linalg.norm(tau_self) * 0.006)
        self._self_fuel -= self_burn

        # ── target fuel & MASS reduction ────────────────────────────────
        # Fuel burned (kg) = k_mass * ||tau_target|| * dt
        tau_mag = float(np.linalg.norm(tau_target))
        fuel_burned_kg = self.FUEL_K_MASS * tau_mag * self.CTL_DT
        self._fuel_mass = max(0.0, self._fuel_mass - fuel_burned_kg)
        self._total_mass = self._dry_mass + self._fuel_mass

        # Update target_fuel (normalized for terminal check)
        self._target_fuel = self._fuel_mass / max(self._fuel_mass_init, 1e-6)

        # ── update MJCF target body mass & inertia ──────────────────────
        bid = self._target_bid
        self.model.body_mass[bid] = self._total_mass
        self.model.body_inertia[bid] = self._orig_inertia_ratio * self._total_mass
        # Re-forward to update cached derived quantities
        mujoco.mj_forward(self.model, self.data)

        # ── torque tracking ────────────────────────────────────────────
        self._tau_history.append(tau_mag)
        if len(self._tau_history) > 30:
            self._tau_history.pop(0)
        self._tau_ma = np.mean(self._tau_history) if self._tau_history else tau_mag
        self._prev_tau_mag = tau_mag

        # Cumulative target torque (for mass estimation proxy)
        # Only accumulate while fuel remains — once depleted, mass stops changing
        if self._fuel_mass > 0:
            self._cumulative_target_torque += tau_mag * self.CTL_DT

        # ── angular acceleration ───────────────────────────────────────
        self._alpha = (omega_new - self._prev_omega) / self.CTL_DT
        self._prev_omega = omega_new.copy()

        # ── mass estimation: cumulative torque-based proxy ───────────────
        # fuel_burned_est = cum_torque * k; est_mass = initial_guess - fuel_burned
        # The SAC network learns to correct k and initial_guess via mass_loss
        cum_torque = self._cumulative_target_torque
        fuel_burned_est = cum_torque * self._mass_fuel_k
        # initial_guess starts at total_mass, network bias-corrects over time
        if not hasattr(self, '_mass_initial_guess'):
            self._mass_initial_guess = self._total_mass
        self._est_mass = self._mass_initial_guess - fuel_burned_est
        self._est_mass_ema = 0.9 * self._est_mass_ema + 0.1 * self._est_mass

        # ── phase switch: mass stabilized → dry_mass reached ────────────
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

        # ── reward ─────────────────────────────────────────────────────
        reward = self._compute_reward(target_burn=tau_mag * self.CTL_DT * 0.006,
                                      self_burn=self_burn,
                                      att_err=sigma_err,
                                      action=action_clipped)

        # Mass estimation loss: encourage agent to estimate dry_mass
        mass_error = (self._est_mass - self._dry_mass)
        r_mass = -self.W_MASS * (mass_error / max(self._dry_mass, 1.0)) ** 2
        reward += r_mass

        # Phase switch bonus (once per episode)
        if self._phase_switched and not self._bonus_given:
            reward += self.R_DETECT
            self._bonus_given = True

        # ── terminal ───────────────────────────────────────────────────
        terminated = False
        truncated  = self._step_count >= self.max_steps
        att_err    = float(np.linalg.norm(sigma_err))

        if self._fuel_mass <= 0.0:
            # Fuel depleted — trigger phase switch, continue for attitude hold
            if not self._phase_switched:
                self._phase_switched = True
                reward += self.R_SUCCESS  # still reward successful depletion
            # Don't terminate — continue episode for attitude maintenance
            self._fuel_mass = 0.0
            self._total_mass = self._dry_mass
        elif self._self_fuel <= 0.0:
            terminated = True; reward += self.R_FAIL
        elif att_err > self.MAX_ATT_ERR:
            terminated = True; reward += self.R_FAIL

        self._prev_action = action_clipped.copy()
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
            # ── true values (hidden from agent! for logging only) ──
            "dry_mass": self._dry_mass,
            "fuel_mass": self._fuel_mass,
            "total_mass": self._total_mass,
            # ── estimated values ──
            "est_mass": self._est_mass,
            "mass_error": mass_error,
            # ── phase switch ──
            "phase_switched": self._phase_switched,
            "mass_stable_ctr": self._mass_stable_ctr,
        }

    # ── observation ────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        sigma = self._get_mrp()
        omega = self.data.qvel[0:3].copy()
        att_err = float(np.linalg.norm(sigma))
        tau_ma  = self._tau_ma
        t_norm  = min(self._step_count / max(self.max_steps, 1), 1.0)

        # Mass estimation features (normalized for network)
        est_mass_norm = self._est_mass / 800.0        # ~ [0.5, 1.0]
        if len(self._mass_history) >= 10:
            mass_change_rate = abs(self._est_mass - self._mass_history[-10]) / (
                10 * self.CTL_DT + 1e-6)
        else:
            mass_change_rate = 0.0
        mass_change_norm = min(1.0, mass_change_rate / 5.0)  # clip to [0,1]
        dmass_detection = min(1.0, self._mass_stable_ctr / self.MASS_STABLE_STEPS)

        return np.array([
            sigma[0], sigma[1], sigma[2],
            omega[0], omega[1], omega[2],
            float(np.clip(self._self_fuel, 0.0, 1.0)),
            est_mass_norm,
            mass_change_norm,
            tau_ma / max(tau_ma + 5.0, 1.0),
            dmass_detection,
            t_norm,
            att_err,
        ], dtype=np.float32)

    def _get_mrp(self) -> np.ndarray:
        return quat_to_mrp(self.data.qpos[3:7].copy())

    def _compute_reward(self, target_burn, self_burn, att_err, action):
        excess = max(0.0, target_burn - self.BASELINE_BURN)
        return (self.W_FUEL * excess
                - self.W_SELF * self_burn
                - self.W_LYAP * float(np.linalg.norm(att_err))
                - self.W_SMOOTH * float(np.linalg.norm(action - self._prev_action)))

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
