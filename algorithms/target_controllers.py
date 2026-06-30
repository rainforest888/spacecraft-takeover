# algorithms/target_controllers.py
"""Adversarial target spacecraft controllers: PID, SMC, LQR variants."""
import numpy as np
from scipy.linalg import solve_continuous_are


class PIDController:
    """PID attitude controller used as adversarial target strategy."""
    def __init__(self, kp: float, ki: float, kd: float, max_torque: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_torque = max_torque
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def reset(self):
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def compute(self, attitude_error: np.ndarray, angular_vel: np.ndarray,
                dt: float) -> np.ndarray:
        self._integral += attitude_error * dt
        derivative = (attitude_error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = attitude_error.copy()
        tau = (self.kp * attitude_error
               + self.ki * self._integral
               + self.kd * derivative)
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        return tau


class SMCController:
    """Sliding mode attitude controller for adversarial target."""
    def __init__(self, lambda_: float, eta: float, max_torque: float):
        self.lambda_ = lambda_
        self.eta = eta
        self.max_torque = max_torque

    def reset(self):
        pass

    def compute(self, attitude_error: np.ndarray, angular_vel: np.ndarray,
                dt: float) -> np.ndarray:
        s = angular_vel + self.lambda_ * attitude_error
        tau = -self.eta * np.sign(s)
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        return tau


class TargetLQRController:
    """LQR attitude controller used as adversarial target."""
    def __init__(self, Q_diag: tuple, R_diag: tuple, max_torque: float):
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.max_torque = max_torque
        self._K = None
        self._I = None

    def set_inertia(self, I_body: np.ndarray):
        """Set inertia and precompute LQR gains."""
        self._I = I_body
        inv_I = np.linalg.inv(I_body)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = 0.5 * np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, :] = inv_I
        P = solve_continuous_are(A, B, self.Q, self.R)
        self._K = np.linalg.inv(self.R) @ B.T @ P

    def reset(self):
        pass

    def compute(self, attitude_error: np.ndarray, angular_vel: np.ndarray,
                dt: float) -> np.ndarray:
        if self._K is None:
            return np.clip(-5.0 * attitude_error - 2.0 * angular_vel,
                           -self.max_torque, self.max_torque)
        x = np.concatenate([attitude_error, angular_vel])
        tau = -self._K @ x
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        return tau


def make_target_strategies() -> list:
    """Create 10 distinct target strategies for adversarial training."""
    strategies = []
    # PID: 3 variants (target weaker than agent: max_torque=12 vs agent 15)
    for kp in [2.0, 5.0, 8.0]:
        for ki in [0.1, 0.5]:
            if len(strategies) < 3:
                strategies.append(PIDController(kp=kp, ki=ki, kd=kp*0.5, max_torque=12.0))
    # SMC: 3 variants
    for lam in [1.0, 2.0, 3.0]:
        if len(strategies) < 6:
            strategies.append(SMCController(lambda_=lam, eta=5.0, max_torque=12.0))
    # LQR: 4 variants
    lqr_configs = [
        ((30, 30, 30, 5, 5, 5), (0.5, 0.5, 0.5)),
        ((50, 50, 50, 10, 10, 10), (1.0, 1.0, 1.0)),
        ((20, 20, 20, 3, 3, 3), (0.2, 0.2, 0.2)),
        ((100, 100, 100, 20, 20, 20), (2.0, 2.0, 2.0)),
    ]
    for Qd, Rd in lqr_configs:
        if len(strategies) < 10:
            strategies.append(TargetLQRController(Q_diag=Qd, R_diag=Rd, max_torque=12.0))
    return strategies[:10]
