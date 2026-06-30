# tests/test_env.py
import numpy as np
from envs.dynamics import (
    quat_to_mrp, mrp_to_quat,
    skew_symmetric,
    compute_gravity_gradient_torque,
    compute_combined_inertia,
    estimate_target_torque,
    fuel_consumed_this_step,
    mrp_error,
)


def test_skew_symmetric():
    v = np.array([1.0, 2.0, 3.0])
    S = skew_symmetric(v)
    assert S.shape == (3, 3)
    assert S[0, 0] == 0 and S[1, 1] == 0 and S[2, 2] == 0
    assert S[0, 1] == -v[2] and S[1, 0] == v[2]
    assert np.allclose(S @ v, np.zeros(3))


def test_mrp_quat_roundtrip():
    np.random.seed(42)
    for _ in range(100):
        mrp = np.random.uniform(-1, 1, 3)
        q = mrp_to_quat(mrp)
        mrp_back = quat_to_mrp(q)
        assert np.allclose(mrp_back, mrp, atol=1e-6) or np.allclose(mrp_back, -mrp, atol=1e-6)


def test_mrp_error_zero():
    mrp_current = np.array([0.1, -0.2, 0.05])
    mrp_desired = mrp_current.copy()
    err = mrp_error(mrp_current, mrp_desired)
    assert np.allclose(err, np.zeros(3), atol=1e-10)


def test_gravity_gradient_zero_when_r_parallel_to_z():
    I = np.diag([500.0, 541.667, 541.667])
    r_hat = np.array([0.0, 0.0, 1.0])
    tau = compute_gravity_gradient_torque(I, r_hat)
    assert np.allclose(tau, np.zeros(3), atol=1e-6)


def test_gravity_gradient_nonzero():
    I = np.diag([500.0, 541.667, 541.667])
    # r_hat not aligned with a principal axis => nonzero torque
    r_hat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    tau = compute_gravity_gradient_torque(I, r_hat)
    assert not np.allclose(tau, np.zeros(3), atol=1e-6)


def test_fuel_consumed():
    torque = np.array([1.0, 2.0, 3.0])
    dt = 0.016
    k = 0.001
    consumed = fuel_consumed_this_step(torque, dt, k)
    expected = dt * np.linalg.norm(torque) * k
    assert abs(consumed - expected) < 1e-10
    assert consumed > 0


def test_estimate_target_torque():
    I = np.diag([100.0, 100.0, 100.0])
    omegadot = np.array([0.1, -0.05, 0.02])
    omega = np.array([0.5, -0.3, 0.1])
    tau_self = np.array([1.0, 0.0, -0.5])
    tau_gg = np.array([0.01, 0.0, 0.0])
    tau_target = estimate_target_torque(I, omegadot, omega, tau_self, tau_gg)
    expected = I @ omegadot + np.cross(omega, I @ omega) - tau_self - tau_gg
    assert np.allclose(tau_target, expected, atol=1e-6)


def test_combined_inertia_parallel_axis():
    I1 = np.diag([20.0, 20.0, 20.0])
    I2 = np.diag([50.0, 50.0, 50.0])
    m1, m2 = 100.0, 500.0
    r1, r2 = np.zeros(3), np.array([2.5, 0.0, 0.0])
    I = compute_combined_inertia(I1, I2, m1, m2, r1, r2)
    assert I.shape == (3, 3)
    # parallel axis: displacement along x, so Ixx unchanged; Iyy, Izz increase
    assert I[0, 0] >= I1[0, 0] + I2[0, 0]


from algorithms.target_controllers import PIDController, SMCController, TargetLQRController


def test_pid_controller_output_shape():
    pid = PIDController(kp=2.0, ki=0.1, kd=1.0, max_torque=10.0)
    attitude_error = np.array([0.1, -0.2, 0.05])
    angular_vel = np.array([0.01, -0.02, 0.0])
    tau = pid.compute(attitude_error, angular_vel, dt=0.016)
    assert tau.shape == (3,)
    assert np.all(np.abs(tau) <= 10.0 + 1e-6)


def test_pid_integral_accumulates():
    pid = PIDController(kp=1.0, ki=1.0, kd=0.0, max_torque=100.0)
    error = np.array([1.0, 0.0, 0.0])
    t1 = pid.compute(error, np.zeros(3), dt=0.1)
    t2 = pid.compute(error, np.zeros(3), dt=0.1)
    assert not np.allclose(t1, t2)
    pid.reset()


def test_smc_controller_output():
    smc = SMCController(lambda_=1.0, eta=2.0, max_torque=10.0)
    error = np.array([0.5, -0.3, 0.1])
    omega = np.array([0.1, 0.0, -0.05])
    tau = smc.compute(error, omega, dt=0.016)
    assert tau.shape == (3,)
    assert np.all(np.abs(tau) <= 10.0 + 1e-6)


def test_target_lqr_controller_output():
    lqr = TargetLQRController(Q_diag=(30, 30, 30, 5, 5, 5),
                               R_diag=(0.5, 0.5, 0.5),
                               max_torque=15.0)
    error = np.array([0.2, -0.1, 0.05])
    omega = np.array([0.05, -0.03, 0.01])
    tau = lqr.compute(error, omega, dt=0.016)
    assert tau.shape == (3,)
    assert np.all(np.abs(tau) <= 15.0 + 1e-6)


import sys
sys.path.insert(0, 'G:/claude code_workspace/spacecraft-takeover')
from envs.spacecraft_env import SpacecraftTakeoverEnv


def test_env_reset_returns_valid_obs():
    env = SpacecraftTakeoverEnv(render_mode=None)
    obs, info = env.reset()
    assert obs.shape == (env.observation_space.shape[0],)
    assert -np.inf < obs[7] < np.inf  # τ_target_mag
    assert 0.0 <= obs[6] <= 1.0       # f_self (index 6 in 11-dim obs)
    env.close()


def test_env_step_returns_correct_shapes():
    env = SpacecraftTakeoverEnv(render_mode=None)
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info['self_fuel'] >= 0.0
    env.close()


def test_env_episode_terminates_on_truncation():
    env = SpacecraftTakeoverEnv(render_mode=None, max_steps=50)
    obs, _ = env.reset()
    step_count = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated
    assert step_count <= 50
    env.close()


def test_env_observation_bounds():
    env = SpacecraftTakeoverEnv(render_mode=None)
    for _ in range(3):
        obs, _ = env.reset()
        # f_self check
        assert 0.0 <= obs[6] <= 1.0
        # t_elapsed check
        assert 0.0 <= obs[10] <= 1.0
        for __ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
    env.close()


def test_env_exposes_physics_state():
    env = SpacecraftTakeoverEnv(render_mode=None)
    obs, _ = env.reset()
    assert env.data is not None
    env.close()
