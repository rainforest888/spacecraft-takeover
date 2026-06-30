# envs/dynamics.py
"""Spacecraft dynamics utilities: MRP, Euler, gravity gradient, fuel model."""
import numpy as np

MU_EARTH = 3.986e14       # m³/s²
R_LEO = 6.8e6              # 400 km altitude


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to MRP [σ₁, σ₂, σ₃]."""
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:4]
    denom = 1.0 + w
    if abs(denom) < 1e-10:
        return np.zeros(3)
    return xyz / denom


def mrp_to_quat(sigma: np.ndarray) -> np.ndarray:
    """Convert MRP [σ₁, σ₂, σ₃] to quaternion [w, x, y, z]."""
    s2 = np.dot(sigma, sigma)
    denom = 1.0 + s2
    w = (1.0 - s2) / denom
    xyz = 2.0 * sigma / denom
    q = np.array([w, xyz[0], xyz[1], xyz[2]])
    return q / np.linalg.norm(q)


def mrp_error(sigma_current: np.ndarray, sigma_desired: np.ndarray) -> np.ndarray:
    """Compute MRP attitude error via quaternion composition."""
    q_c = mrp_to_quat(sigma_current)
    q_d = mrp_to_quat(sigma_desired)
    q_d_inv = np.array([q_d[0], -q_d[1], -q_d[2], -q_d[3]])
    w1, x1, y1, z1 = q_d_inv
    w2, x2, y2, z2 = q_c
    q_err = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return quat_to_mrp(q_err)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric cross-product matrix of v."""
    return np.array([
        [0,      -v[2],   v[1]],
        [v[2],    0,     -v[0]],
        [-v[1],   v[0],   0   ],
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q. Returns 3-vector."""
    w, x, y, z = q
    qv = np.array([-x*v[0]-y*v[1]-z*v[2], w*v[0]+y*v[2]-z*v[1],
                    w*v[1]+z*v[0]-x*v[2], w*v[2]+x*v[1]-y*v[0]])
    q_conj = np.array([w, -x, -y, -z])
    w1, x1, y1, z1 = qv
    w2, x2, y2, z2 = q_conj
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def compute_gravity_gradient_torque(I_body: np.ndarray, r_hat_body: np.ndarray,
                                     mu: float = MU_EARTH, R: float = R_LEO) -> np.ndarray:
    """Gravity gradient torque: τ_gg = (3μ/R³)(r̂ × I·r̂)."""
    gg_coeff = 3.0 * mu / (R ** 3)
    Ir = I_body @ r_hat_body
    return gg_coeff * np.cross(r_hat_body, Ir)


def compute_combined_inertia(I1: np.ndarray, I2: np.ndarray,
                              m1: float, m2: float,
                              r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Combined-body inertia about system CoM using parallel axis theorem."""
    M = m1 + m2
    r_cm = (m1 * r1 + m2 * r2) / M
    r1_rel = r1 - r_cm
    r2_rel = r2 - r_cm
    d1_sq = np.dot(r1_rel, r1_rel) * np.eye(3) - np.outer(r1_rel, r1_rel)
    d2_sq = np.dot(r2_rel, r2_rel) * np.eye(3) - np.outer(r2_rel, r2_rel)
    return I1 + I2 + m1 * d1_sq + m2 * d2_sq


def fuel_consumed_this_step(torque: np.ndarray, dt: float, k_consumption: float) -> float:
    """Fuel consumed: Δf = dt * ||τ|| * k."""
    return float(dt * np.linalg.norm(torque) * k_consumption)


def estimate_target_torque(I_combined: np.ndarray, omegadot: np.ndarray,
                            omega: np.ndarray, tau_self: np.ndarray,
                            tau_disturbance: np.ndarray) -> np.ndarray:
    """Estimate target torque from rigid-body dynamics.
    τ_target_est = I·ω̇ + ω×(I·ω) - τ_self - τ_disturbance.
    """
    tau_total_obs = I_combined @ omegadot + np.cross(omega, I_combined @ omega)
    return tau_total_obs - tau_self - tau_disturbance
