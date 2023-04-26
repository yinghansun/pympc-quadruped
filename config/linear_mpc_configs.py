from typing import List
import numpy as np

class LinearMpcConfig:

    dt_control: float = 0.001
    iteration_between_mpc: int = 20
    # dt_mpc: float = dt_control * iteration_between_mpc
    dt_mpc: float = 0.05

    horizon: int = 16

    gravity: np.float16 = 9.81

    friction_coef: float = 0.7

    # quadratic programming weights
    # r, p, y, x, y, z, wx, wy, wz, vx, vy, vz, g
    Q: np.ndarray = np.diag([5., 5., 10., 10., 10., 50., 0.01, 0.01, 0.2, 0.2, 0.2, 0.2, 0.])
    R: np.ndarray = np.diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])

    cmd_xvel: float = 0.
    cmd_yvel: float = 0.
    cmd_yaw_turn_rate: float = 0.