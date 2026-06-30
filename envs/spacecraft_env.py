# envs/spacecraft_env.py
"""Gymnasium environment for spacecraft attitude takeover using MuJoCo physics."""
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

    # Reward weights (from spec)
    W1 = 10.0   # target fuel consumed
    W2 = 5.0    # self fuel cost
    W3 = 2.0    # attitude error
    W4 = 1.0    # angular velocity
    W5 = 3.0    # burst bonus
    W6 = 2.0    # weakening bonus

    R_SUCCESS = 100.0
    R_FAIL = -100.0

    MAX_ATTITUDE_ERROR = np.pi
    MAX_TORQUE = 10.0
    FUEL_K = 0.005
    INITIAL_FUEL = 1.0

    CTL_DT = 1.0 / 60.0          # 60 Hz control
    SUBSTEPS = 30                 # MuJoCo substeps per control step

    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Load model & data
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.CTL_DT / self.SUBSTEPS

        # Observation space: 11-dim
        obs_high = np.array([
            np.inf, np.inf, np.inf,       # MRP
            np.inf, np.inf, np.inf,       # angular velocity
            1.0,                          # f_self
            np.inf,                       # τ_target_mag
            np.inf,                       # τ_target_mag_ma
            np.inf,                       # Δτ_target
            1.0,                          # t_elapsed
        ], dtype=np.float32)
        obs_low = np.array([
            -np.inf, -np.inf, -np.inf,
            -np.inf, -np.inf, -np.inf,
            0.0,
            0.0, 0.0,
            -np.inf,
            0.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: 3-dim torque [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Target strategies
        self.target_strategies = make_target_strategies()
        self._current_target_strategy = None

        # Internal state
        self._step_count = 0
        self._self_fuel = 1.0
        self._target_fuel = 1.0
        self._tau_target_history = []
        self._peak_torque_observed = 0.0
        self._tau_target_ma = 0.0
        self._prev_tau_target_mag = 0.0

    # --- Gym interface ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random if self.np_random is not None else np.random

        # Random initial attitude
        sigma_init = rng.uniform(-0.5, 0.5, 3)
        q_init = self._mrp_to_quat_np(sigma_init)
        self.data.qpos[3:7] = q_init
        omega_init = rng.uniform(-0.2, 0.2, 3)
        self.data.qvel[0:3] = omega_init
        mujoco.mj_forward(self.model, self.data)

        # Reset internal state
        self._step_count = 0
        self._self_fuel = self.INITIAL_FUEL
        self._target_fuel = self.INITIAL_FUEL
        self._tau_target_history = []
        self._peak_torque_observed = 0.0
        self._tau_target_ma = 0.0
        self._prev_tau_target_mag = 0.0

        # Pick target strategy
        idx = rng.integers(0, len(self.target_strategies))
        self._current_target_strategy = self.target_strategies[idx]
        if hasattr(self._current_target_strategy, 'reset'):
            self._current_target_strategy.reset()
        if hasattr(self._current_target_strategy, 'set_inertia'):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sat")
            I_diag = self.model.body_inertia[body_id]
            I = np.diag(I_diag)
            self._current_target_strategy.set_inertia(I)

        obs = self._get_obs()
        info = {"self_fuel": self._self_fuel, "target_fuel": self._target_fuel,
                "target_strategy": type(self._current_target_strategy).__name__}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        tau_self = np.clip(action, -1.0, 1.0) * self.MAX_TORQUE

        # Get state for target controller
        omega = self.data.qvel[0:3].copy()
        sigma_current = self._get_current_mrp()
        sigma_err_target = mrp_error(sigma_current, np.zeros(3))

        # Target computes resisting torque
        tau_target = self._current_target_strategy.compute(sigma_err_target, omega, dt=self.CTL_DT)

        # Gravity gradient
        I_comb = self._get_combined_inertia()
        r_hat = np.array([1.0, 0.0, 0.0])
        tau_gg = compute_gravity_gradient_torque(I_comb, r_hat)

        tau_total = tau_self + tau_target + tau_gg

        # Apply torque and step physics
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        for _ in range(self.SUBSTEPS):
            self.data.xfrc_applied[body_id, 0:3] = tau_total
            self.data.ctrl[0:3] = tau_total
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)

        # Fuel
        self_fuel_used = fuel_consumed_this_step(tau_self, self.CTL_DT, self.FUEL_K)
        target_fuel_used = fuel_consumed_this_step(tau_target, self.CTL_DT, self.FUEL_K)
        self._self_fuel -= self_fuel_used
        self._target_fuel -= target_fuel_used

        # Track target torque
        tau_target_mag = float(np.linalg.norm(tau_target))
        self._tau_target_history.append(tau_target_mag)
        if len(self._tau_target_history) > 30:
            self._tau_target_history.pop(0)
        self._tau_target_ma = np.mean(self._tau_target_history)
        self._peak_torque_observed = max(self._peak_torque_observed, tau_target_mag)
        delta_tau = tau_target_mag - self._prev_tau_target_mag
        self._prev_tau_target_mag = tau_target_mag

        # Reward
        reward = self._compute_reward(target_fuel_used, self_fuel_used,
                                       sigma_err_target, omega,
                                       target_fuel_used)

        # Terminal
        terminated = False
        truncated = self._step_count >= self.max_steps
        att_err = float(np.linalg.norm(sigma_err_target))

        if self._target_fuel <= 0:
            terminated = True
            reward += self.R_SUCCESS
        elif self._self_fuel <= 0:
            terminated = True
            reward += self.R_FAIL
        elif att_err > self.MAX_ATTITUDE_ERROR:
            terminated = True
            reward += self.R_FAIL

        obs = self._get_obs()
        info = {
            "self_fuel": self._self_fuel,
            "target_fuel": self._target_fuel,
            "tau_target_mag": tau_target_mag,
            "attitude_error": att_err,
            "reward": reward,
            "target_strategy": type(self._current_target_strategy).__name__,
            "self_fuel_used": self_fuel_used,
            "target_fuel_used": target_fuel_used,
        }
        return obs, reward, terminated, truncated, info

    # --- Observation ---
    def _get_obs(self):
        sigma = self._get_current_mrp()
        omega = self.data.qvel[0:3].copy()
        tau_mag = self._prev_tau_target_mag
        tau_ma = self._tau_target_ma
        delta_tau = (self._tau_target_history[-2] - tau_mag
                     if len(self._tau_target_history) >= 2 else 0.0)
        t_elapsed = min(self._step_count / max(self.max_steps, 1), 1.0)
        return np.array([
            sigma[0], sigma[1], sigma[2],
            omega[0], omega[1], omega[2],
            float(np.clip(self._self_fuel, 0.0, 1.0)),
            tau_mag,
            tau_ma,
            delta_tau,
            t_elapsed,
        ], dtype=np.float32)

    def _get_current_mrp(self):
        q = self.data.qpos[3:7].copy()
        return quat_to_mrp(q)

    def _get_combined_inertia(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        I_diag = self.model.body_inertia[body_id]
        return np.diag(I_diag)

    def _compute_reward(self, target_fuel_used, self_fuel_used,
                         attitude_err, omega, target_burn):
        r_target = self.W1 * target_fuel_used
        r_self = -self.W2 * self_fuel_used
        r_att = -self.W3 * float(np.linalg.norm(attitude_err))
        r_omg = -self.W4 * float(np.linalg.norm(omega))
        r_burst = self.W5 * max(0.0, target_burn - 0.01)
        r_weak = self.W6 * max(0.0, self._tau_target_ma - self._prev_tau_target_mag)
        return r_target + r_self + r_att + r_omg + r_burst + r_weak

    def _mrp_to_quat_np(self, sigma):
        s2 = np.dot(sigma, sigma)
        denom = 1.0 + s2
        w = (1.0 - s2) / denom
        xyz = 2.0 * sigma / denom
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 640, 480)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        pass
