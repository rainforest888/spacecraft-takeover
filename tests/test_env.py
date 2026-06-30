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
