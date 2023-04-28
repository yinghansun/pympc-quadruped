import numpy as np

def make_com_inertial_matrix(
    ixx: float, ixy: float, ixz: float, iyy: float, iyz: float, izz: float
) -> np.ndarray:
    '''Compute the inertia matrix expressed in the link's CoM frame 
    (from URDF).

    Args
    ----
    six numbers (ixx, ixy, ixz, iyy, iyz, izz) in the URDF file corresponding 
    to the inertia matrix.
    '''
    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz]
    ], dtype=np.float32)