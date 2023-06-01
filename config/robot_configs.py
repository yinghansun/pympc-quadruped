import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from dynamics import make_com_inertial_matrix

class RobotConfig:
    mass_base: float
    base_height_des: float
    base_inertia_base: np.ndarray

    fz_max: float
    
    swing_height: float
    Kp_swing: np.ndarray
    Kd_swing: np.ndarray


class AliengoConfig(RobotConfig):
    mass_base: float = 9.042
    base_height_des: float = 0.38
    base_inertia_base = make_com_inertial_matrix(
        ixx=0.033260231, 
        ixy=-0.000451628, 
        ixz=0.000487603, 
        iyy=0.16117211, 
        iyz=4.8356e-05, 
        izz=0.17460442
    )

    fz_max = 500.

    swing_height = 0.1
    Kp_swing = np.diag([200., 200., 200.])
    Kd_swing = np.diag([20., 20., 20.])


class A1Config(RobotConfig):
    mass_base: float = 4.713
    base_height_des: float = 0.42
    base_inertia_base = make_com_inertial_matrix(
        ixx=0.01683993, 
        ixy=8.3902e-05, 
        ixz=0.000597679, 
        iyy=0.056579028, 
        iyz=2.5134e-05, 
        izz=0.064713601
    ) * 10

    fz_max = 500.

    swing_height = 0.1
    Kp_swing = np.diag([700., 700., 700.])
    Kd_swing = np.diag([20., 20., 20.])