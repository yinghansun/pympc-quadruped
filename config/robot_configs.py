import numpy as np

class RobotConfig:
    mass_base: float
    base_height_des: float

    fz_max: float
    
    swing_height: float
    Kp_swing: np.ndarray
    Kd_swing: np.ndarray


class AliengoConfig(RobotConfig):
    mass_base: float = 9.042
    base_height_des: float = 0.42
    
    fz_max = 500.

    swing_height = 0.1
    Kp_swing = np.diag([200., 200., 200.])
    Kd_swing = np.diag([20., 20., 20.])