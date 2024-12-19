"""Common utility functions."""

import numpy as np


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands

    Args:
        target_q: The desired state.
        q: The current state.
        kp: The proportional gain.
        dq: The desired state.
        kd: The derivative gain.
        default: The default state.
    """
    return kp * (target_q + default - q) - kd * dq


def quaternion_to_euler_array(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to a 3D Euler angle array.

    Args:
        quat: The quaternion to convert, with shape `(4, *)`

    Returns:
        The Euler angles, with shape `(3, *)`
    """
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])
