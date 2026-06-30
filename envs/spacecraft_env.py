"""Gymnasium environment for spacecraft attitude takeover using MuJoCo physics.

Core task: a small service spacecraft (100 kg, MAX_TORQUE=15 Nm) is welded to
a large non-cooperative target (500 kg).  The target actively resists using a
randomly-selected PID/SMC/LQR controller.

The agent must learn to apply torque that *provokes* the target into burning
fuel while conserving its own—then stabilise the combined body once the target
is exhausted.

Design principles
-----------------
* Equal fuel rate per N·m for both bodies (FUEL_K = 0.008).  The agent's
  tactical advantage comes from choosing *when* and *how* to move, forcing
  the target into high-torque, fuel-wasting responses.
* Reward is shaped around *fuel efficiency* (target_burn − self_burn)
  multiplied by a large scale so the per-step signal is easily visible.
* A small survival bonus (+0.05/step) encourages the agent to stay alive
  long enough to discover the fuel-advantage signal.
* Terminal penalties are deliberately smaller than the spec's ±100 so early
  exploration failures don't drown out all learning signal.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from envs.dynamics import (
    quat_to_mrp, mrp_error,
    compute_gravity_gradient_torque, fuel_consumed_this_step,
)
from algorithms.target_controllers import make_target_strategies

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "combo_body.xml")


class SpacecraftTakeoverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ── reward shaping ────────────────────────────────────────────────
    R_SURVIVE    = 0.1      # small per-step survival bonus

    W_ADVANTAGE  = 200.0    # (target_burn − self_burn) * 200 → ~0.4/step signal
    W_ATTITUDE   = 0.5      # very soft attitude penalty — let agent explore
    W_OMEGA      = 0.2      # very soft rate penalty
    W_BURST      = 1.0      # short-window burst bonus
    W_WEAKENING  = 1.0      # target torque dropping bonus

    R_SUCCESS    = 200.0     # target exhausted — BIG reward
    R_FAIL       = -50.0     # self exhausted or tumble

    # ── physical parameters ────────────────────────────────────────────
    MAX_TORQUE         = 15.0    # N·m  (service sat)
    FUEL_K             = 0.02    # fuel / (N·m·s) — faster depletion, ~430 steps to exhaust
    INITIAL_FUEL       = 1.0
    MAX_ATTITUDE_ERROR = np.pi   # rad — tumble threshold

    CTL_DT   = 1.0 / 60.0        # 60 Hz control
    SUBSTEPS = 30                 # MuJoCo substeps per control step

    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = self.CTL_DT / self.SUBSTEPS

        # ── observation: 11 dim ────────────────────────────────────────
        obs_high = np.array(
            [np.inf, np.inf, np.inf,      # MRP attitude
             np.inf, np.inf, np.inf,      # angular velocity
             1.0,                          # f_self  [0, 1]
             np.inf, np.inf,              # τ_target_mag, τ_target_ma
             np.inf,                      # Δτ_target
             1.0],                         # t_elapsed [0, 1]
            dtype=np.float32)
        obs_low = np.array(
            [-np.inf, -np.inf, -np.inf,
             -np.inf, -np.inf, -np.inf,
             0.0,
             0.0, 0.0,
             -np.inf,
             0.0],
            dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ── action: 3-dim torque, normalised to [-1, 1] ───────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # adversarial targets
        self.target_strategies = make_target_strategies()
        self._current_strategy = None

        # internal state
        self._step_count          = 0
        self._self_fuel           = 1.0
        self._target_fuel         = 1.0
        self._tau_history         = []      # fixed-length ring buffer
        self._peak_torque         = 0.0
        self._tau_ma              = 0.0
        self._prev_tau_mag        = 0.0
        self._target_burn_window  = []      # last 10 steps of target fuel used

    # ── Gym API ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random if self.np_random is not None else np.random

        # random initial attitude & angular velocity
        sigma_init = rng.uniform(-0.15, 0.15, 3)
        q_init = self._mrp2quat(sigma_init)
        self.data.qpos[3:7] = q_init
        self.data.qvel[0:3] = rng.uniform(-0.1, 0.1, 3)
        mujoco.mj_forward(self.model, self.data)

        # reset counters
        self._step_count  = 0
        self._self_fuel   = self.INITIAL_FUEL
        self._target_fuel = self.INITIAL_FUEL
        self._tau_history.clear()
        self._peak_torque   = 0.0
        self._tau_ma        = 0.0
        self._prev_tau_mag  = 0.0
        self._target_burn_window.clear()

        # pick adversarial controller
        idx = rng.integers(0, len(self.target_strategies))
        self._current_strategy = self.target_strategies[idx]
        if hasattr(self._current_strategy, 'reset'):
            self._current_strategy.reset()
        if hasattr(self._current_strategy, 'set_inertia'):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sat")
            I = np.diag(self.model.body_inertia[bid])
            self._current_strategy.set_inertia(I)

        obs = self._get_obs()
        return obs, {"self_fuel": self._self_fuel, "target_fuel": self._target_fuel,
                     "target_strategy": type(self._current_strategy).__name__}

    def step(self, action: np.ndarray):
        self._step_count += 1
        tau_self = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * self.MAX_TORQUE

        # ── target response ────────────────────────────────────────────
        omega     = self.data.qvel[0:3].copy()
        sigma_cur = self._get_mrp()
        sigma_err = mrp_error(sigma_cur, np.zeros(3))            # target wants σ=0
        tau_target = self._current_strategy.compute(sigma_err, omega, dt=self.CTL_DT)

        # ── gravity gradient ───────────────────────────────────────────
        I_comb = self._get_combined_inertia()
        r_hat  = np.array([1.0, 0.0, 0.0])
        tau_gg = compute_gravity_gradient_torque(I_comb, r_hat)

        # ── apply to MuJoCo ────────────────────────────────────────────
        tau_total = tau_self + tau_target + tau_gg
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        for _ in range(self.SUBSTEPS):
            self.data.xfrc_applied[bid, 0:3] = tau_total
            self.data.ctrl[0:3] = tau_total
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)

        # ── fuel (same k for both!) ────────────────────────────────────
        self_burn   = fuel_consumed_this_step(tau_self,   self.CTL_DT, self.FUEL_K)
        target_burn = fuel_consumed_this_step(tau_target, self.CTL_DT, self.FUEL_K)
        self._self_fuel   -= self_burn
        self._target_fuel -= target_burn

        # ── target-torque tracking ─────────────────────────────────────
        tau_mag = float(np.linalg.norm(tau_target))
        self._tau_history.append(tau_mag)
        if len(self._tau_history) > 30:
            self._tau_history.pop(0)
        self._tau_ma      = np.mean(self._tau_history) if self._tau_history else tau_mag
        self._peak_torque  = max(self._peak_torque, tau_mag)
        delta_tau         = tau_mag - self._prev_tau_mag
        self._prev_tau_mag = tau_mag

        # burst window (target burn in recent 10 steps)
        self._target_burn_window.append(target_burn)
        if len(self._target_burn_window) > 10:
            self._target_burn_window.pop(0)

        # ── reward ─────────────────────────────────────────────────────
        reward = self._compute_reward(target_burn, self_burn, sigma_err, omega, tau_mag)

        # ── terminal checks ────────────────────────────────────────────
        terminated = False
        truncated  = self._step_count >= self.max_steps
        att_err    = float(np.linalg.norm(sigma_err))

        if self._target_fuel <= 0.0:
            terminated = True
            reward    += self.R_SUCCESS
        elif self._self_fuel <= 0.0:
            terminated = True
            reward    += self.R_FAIL
        elif att_err > self.MAX_ATTITUDE_ERROR:
            terminated = True
            reward    += self.R_FAIL

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {
            "self_fuel":       self._self_fuel,
            "target_fuel":     self._target_fuel,
            "tau_target_mag":  tau_mag,
            "attitude_error":  att_err,
            "reward":          reward,
            "target_strategy": type(self._current_strategy).__name__,
            "self_burn":       self_burn,
            "target_burn":     target_burn,
        }

    # ── observation builder ────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        sigma = self._get_mrp()
        omega = self.data.qvel[0:3].copy()
        tau_m = self._prev_tau_mag
        tau_ma = self._tau_ma
        delta  = self._tau_history[-2] - tau_m if len(self._tau_history) >= 2 else 0.0
        t_norm = min(self._step_count / max(self.max_steps, 1), 1.0)
        return np.array([
            sigma[0], sigma[1], sigma[2],
            omega[0], omega[1], omega[2],
            float(np.clip(self._self_fuel, 0.0, 1.0)),
            tau_m,
            tau_ma,
            delta,
            t_norm,
        ], dtype=np.float32)

    def _get_mrp(self) -> np.ndarray:
        return quat_to_mrp(self.data.qpos[3:7].copy())

    def _get_combined_inertia(self) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        return np.diag(self.model.body_inertia[bid])

    # ── reward function ────────────────────────────────────────────────
    def _compute_reward(self, target_burn: float, self_burn: float,
                        att_err: np.ndarray, omega: np.ndarray,
                        tau_mag: float) -> float:
        # efficiency: net fuel advantage per step
        r_adv = self.W_ADVANTAGE * (target_burn - self_burn)

        # attitude / rate penalties
        r_att = -self.W_ATTITUDE * float(np.linalg.norm(att_err))
        r_omg = -self.W_OMEGA   * float(np.linalg.norm(omega))

        # burst: recent target burn exceeds idle baseline (~0.001)
        burst_sum  = sum(self._target_burn_window)
        burst_base = 0.001 * len(self._target_burn_window)
        r_burst    = self.W_BURST * max(0.0, burst_sum - burst_base)

        # weakening: target torque magnitude is dropping
        r_weak = self.W_WEAKENING * max(0.0, self._tau_ma - tau_mag)

        # survival
        r_survive = self.R_SURVIVE

        return r_adv + r_att + r_omg + r_burst + r_weak + r_survive

    # ── rendering ──────────────────────────────────────────────────────
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

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _mrp2quat(sigma: np.ndarray) -> np.ndarray:
        s2 = np.dot(sigma, sigma)
        denom = 1.0 + s2
        w = (1.0 - s2) / denom
        xyz = 2.0 * sigma / denom
        return np.array([w, xyz[0], xyz[1], xyz[2]])
