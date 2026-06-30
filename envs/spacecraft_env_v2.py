"""Gymnasium environment — SAC-optimised version.

Key changes from TD3 version:
1. Gravity gradient torque DISABLED — agent must actively provoke the target
2. Reward: excess target burn over baseline + Lyapunov stability term + smoothness penalty
3. Larger initial perturbations, shorter episode window
4. Stronger PID targets (kp = 10/20/30)
5. PID dead-zone removed — target always responds to error
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
from algorithms.target_controllers import make_target_strategies_strong

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "combo_body.xml")


class SpacecraftTakeoverEnvV2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ── reward weights ─────────────────────────────────────────────────
    # Core insight: fuel signal must dominate Lyapunov penalty
    # target_burn ~0.002, baseline~0.0003 → excess~0.0017 → 500×0.0017=0.85
    # att_err ~0.1 at start → 10×0.01²=0.001 (barely noticeable)
    # att_err ~0.5 (bad) → 10×0.25=2.5 (starts to matter)
    # self fuel cost only matters when agent burns significant fuel
    W_FUEL      = 500.0     # excess target burn above baseline
    W_SELF      = 15.0      # very light self fuel penalty
    W_LYAP      = 10.0      # attitude error penalty (< att_err>0.5 hurts)
    W_SMOOTH    = 3.0       # action smoothness
    R_SUCCESS   = 200.0
    R_FAIL      = -50.0

    # ── physical parameters ────────────────────────────────────────────
    MAX_TORQUE   = 12.0     # slightly below targets to make it harder
    FUEL_K       = 0.006
    INITIAL_FUEL = 1.0
    MAX_ATT_ERR  = np.pi

    CTL_DT   = 1.0 / 60.0
    SUBSTEPS = 30

    BASELINE_BURN = 0.0003   # target burn at rest

    def __init__(self, render_mode=None, max_steps=600):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = self.CTL_DT / self.SUBSTEPS

        # ── observation: 12 dim (adds att_err_norm) ────────────────────
        obs_high = np.array(
            [np.inf]*6 + [1.0, np.inf, np.inf, np.inf, 1.0, 1.0],
            dtype=np.float32)
        obs_low  = np.array(
            [-np.inf]*6 + [0.0, 0.0, 0.0, -np.inf, 0.0, 0.0],
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
        self._peak_torque   = 0.0
        self._tau_ma        = 0.0
        self._prev_tau_mag  = 0.0
        self._prev_action   = np.zeros(3)

    # ── Gym API ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random if self.np_random is not None else np.random

        # Attitude error grows fast even at rest — target can resist ~20°/s
        # before the combined body tumbles. Give ~200 steps of survival.
        # Initial: small error, small rate — survive and act
        sigma_init = rng.uniform(-0.15, 0.15, 3)
        self.data.qpos[3:7] = self._mrp2quat(sigma_init)
        self.data.qvel[0:3] = rng.uniform(-0.1, 0.1, 3)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._self_fuel  = self.INITIAL_FUEL
        self._target_fuel = self.INITIAL_FUEL
        self._tau_history.clear()
        self._peak_torque  = 0.0
        self._tau_ma       = 0.0
        self._prev_tau_mag = 0.0
        self._prev_action  = np.zeros(3)

        idx = rng.integers(0, len(self.target_strategies))
        self._current_strategy = self.target_strategies[idx]
        if hasattr(self._current_strategy, 'reset'):
            self._current_strategy.reset()
        if hasattr(self._current_strategy, 'set_inertia'):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sat")
            self._current_strategy.set_inertia(np.diag(self.model.body_inertia[bid]))

        return self._get_obs(), {
            "self_fuel": self._self_fuel, "target_fuel": self._target_fuel,
            "target_strategy": type(self._current_strategy).__name__}

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

        # ── gravity gradient: DISABLED — agent must actively provoke ────
        tau_gg = np.zeros(3)

        tau_total = tau_self + tau_target + tau_gg
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_sat")
        for _ in range(self.SUBSTEPS):
            self.data.xfrc_applied[bid, 0:3] = tau_total
            self.data.ctrl[0:3] = tau_total
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)

        # ── fuel ───────────────────────────────────────────────────────
        self_burn   = fuel_consumed_this_step(tau_self,   self.CTL_DT, self.FUEL_K)
        target_burn = fuel_consumed_this_step(tau_target, self.CTL_DT, self.FUEL_K)
        self._self_fuel   -= self_burn
        self._target_fuel -= target_burn

        # ── torque tracking ────────────────────────────────────────────
        tau_mag = float(np.linalg.norm(tau_target))
        self._tau_history.append(tau_mag)
        if len(self._tau_history) > 30:
            self._tau_history.pop(0)
        self._tau_ma       = np.mean(self._tau_history) if self._tau_history else tau_mag
        self._peak_torque  = max(self._peak_torque, tau_mag)
        delta_tau          = tau_mag - self._prev_tau_mag
        self._prev_tau_mag = tau_mag

        # ── reward ─────────────────────────────────────────────────────
        reward = self._compute_reward(target_burn, self_burn, sigma_err, action_clipped)

        # ── terminal ───────────────────────────────────────────────────
        terminated = False
        truncated  = self._step_count >= self.max_steps
        att_err    = float(np.linalg.norm(sigma_err))

        if self._target_fuel <= 0.0:
            terminated = True; reward += self.R_SUCCESS
        elif self._self_fuel <= 0.0:
            terminated = True; reward += self.R_FAIL
        elif att_err > self.MAX_ATT_ERR:
            terminated = True; reward += self.R_FAIL

        self._prev_action = action_clipped.copy()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {
            "self_fuel": self._self_fuel, "target_fuel": self._target_fuel,
            "tau_target_mag": tau_mag, "attitude_error": att_err,
            "reward": reward,
            "target_strategy": type(self._current_strategy).__name__,
            "self_burn": self_burn, "target_burn": target_burn,
        }

    # ── observation ────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        sigma = self._get_mrp()
        omega = self.data.qvel[0:3].copy()
        att_err = float(np.linalg.norm(sigma))
        tau_m   = self._prev_tau_mag
        tau_ma  = self._tau_ma
        delta   = self._tau_history[-2] - tau_m if len(self._tau_history) >= 2 else 0.0
        t_norm  = min(self._step_count / max(self.max_steps, 1), 1.0)
        return np.array([
            sigma[0], sigma[1], sigma[2],
            omega[0], omega[1], omega[2],
            float(np.clip(self._self_fuel, 0.0, 1.0)),
            tau_m, tau_ma, delta,
            t_norm,
            att_err,                         # ← extra dim for Lyapunov reward
        ], dtype=np.float32)

    def _get_mrp(self) -> np.ndarray:
        return quat_to_mrp(self.data.qpos[3:7].copy())

    # ── reward function (paper-driven) ─────────────────────────────────
    def _compute_reward(self, target_burn: float, self_burn: float,
                        att_err: np.ndarray, action: np.ndarray) -> float:
        # 1. Excess target burn above passive baseline
        excess = max(0.0, target_burn - self.BASELINE_BURN)
        r_fuel = self.W_FUEL * excess

        # 2. Self fuel cost
        r_self = -self.W_SELF * self_burn

        # 3. Lyapunov stability (SSG-RL inspired):
        #    V = ||σ||², reward decrease in V
        V_now  = float(np.linalg.norm(att_err)) ** 2
        # We don't store V_prev across steps; instead use att_err magnitude
        r_lyap = -self.W_LYAP * float(np.linalg.norm(att_err))   # smaller error = better

        # 4. Action smoothness (Geng et al. 2025)
        r_smooth = -self.W_SMOOTH * float(np.linalg.norm(action - self._prev_action))

        return r_fuel + r_self + r_lyap + r_smooth

    # ── render ─────────────────────────────────────────────────────────
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
    def _mrp2quat(sigma: np.ndarray) -> np.ndarray:
        s2 = np.dot(sigma, sigma)
        denom = 1.0 + s2
        return np.array([(1.0 - s2) / denom,
                         2.0 * sigma[0] / denom,
                         2.0 * sigma[1] / denom,
                         2.0 * sigma[2] / denom])
