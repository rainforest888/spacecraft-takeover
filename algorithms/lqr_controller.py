# algorithms/lqr_controller.py
"""LQR optimal controller for steady-state attitude takeover."""
import numpy as np
from scipy.linalg import solve_continuous_are


class LQRController:
    def __init__(self, I_body: np.ndarray, max_torque: float,
                 Q_diag=(50.0, 50.0, 50.0, 10.0, 10.0, 10.0),
                 R_diag=(1.0, 1.0, 1.0)):
        self.max_torque = max_torque
        # Linearized dynamics about zero attitude:
        # d/dt [σ; ω] = A [σ; ω] + B u
        # A = [[0, 0.5*I₃], [0, 0]]
        # B = [[0], [I⁻¹]]
        inv_I = np.linalg.inv(I_body)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = 0.5 * np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, :] = inv_I

        Q = np.diag(Q_diag)
        R = np.diag(R_diag)
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute(self, attitude_error: np.ndarray, angular_vel: np.ndarray) -> np.ndarray:
        x = np.concatenate([attitude_error, angular_vel])
        tau = -self.K @ x
        return np.clip(tau, -self.max_torque, self.max_torque)
