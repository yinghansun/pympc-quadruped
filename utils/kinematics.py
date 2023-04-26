from typing import Optional, Union
import math
import numpy as np

def vec_format_standardization(
    vec: Union[list, np.ndarray], std_type: Optional[str] = 'list'
) -> Union[list, np.ndarray]:

    assert type(vec) == list or type(vec) == np.ndarray
    assert std_type == 'list' or std_type == '1darray'

    if std_type == list:
        if type(vec) == np.ndarray:
            vec_shape = vec.shape
            assert len(vec_shape) == 1 or vec_shape[0] == 1 or vec_shape[1] == 1

            if len(vec_shape) == 1:
                vec = list(vec)
            elif vec_shape[0] == 1:
                vec = list(vec[0])
            elif vec_shape[1] == 1:
                num_elements = vec_shape[0]
                vec = vec.reshape(1, num_elements)
                vec = list(vec[0])

        return vec

    else:
        if type(vec) == list:
            vec = np.array(vec)
        else:
            vec_shape = vec.shape
            assert len(vec_shape) == 1 or vec_shape[0] == 1 or vec_shape[1] == 1

            if len(vec_shape) != 1:
                vec = vec.flatten()

        return vec

def quat2ZYXangle(quat: Union[list, np.ndarray]) -> list:
    q = vec_format_standardization(quat)
    assert len(q) == 4

    # q[0]:             real part
    # q[1], q[2], q[3]: imaginary part
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return [phi, theta, psi]

def quat2matrix(quat: Union[list, np.ndarray]) -> np.ndarray:
    q = vec_format_standardization(quat)
    assert len(q) == 4

    # q[0]:             real part
    # q[1], q[2], q[3]: imaginary part
    r11 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    r12 = 2 * (q[1] * q[2] - q[0] * q[3])
    r13 = 2 * (q[0] * q[2] + q[1] * q[3])
    r21 = 2 * (q[0] * q[3] + q[1] * q[2])
    r22 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    r23 = 2 * (q[2] * q[3] - q[0] * q[1])
    r31 = 2 * (q[1] * q[3] - q[0] * q[2])
    r32 = 2 * (q[0] * q[1] + q[2] * q[3])
    r33 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    return rotation_matrix

# reference: modern robotics, appendix B.1
def ZYXangles2matrix(ZYX_angle: Union[list, np.ndarray]) -> np.ndarray:
    # ZYX_angle = [phi, theta, psi]
    # R = Rz(psi)Ry(theta)Rx(phi)

    ZYX_angle = vec_format_standardization(ZYX_angle)
    assert len(ZYX_angle) == 3

    phi = ZYX_angle[0]
    theta = ZYX_angle[1]
    psi = ZYX_angle[2]

    r11 = math.cos(psi) * math.cos(theta)
    r12 = math.cos(psi) * math.sin(theta) * math.sin(phi) - math.sin(psi) * math.cos(phi)
    r13 = math.cos(psi) * math.sin(theta) * math.cos(phi) + math.sin(psi) * math.sin(phi)
    r21 = math.sin(psi) * math.cos(theta)
    r22 = math.sin(psi) * math.sin(theta) * math.sin(phi) + math.cos(psi) * math.cos(phi)
    r23 = math.sin(psi) * math.sin(theta) * math.cos(phi) - math.cos(psi) * math.sin(phi)
    r31 = -math.sin(theta)
    r32 = math.cos(theta) * math.sin(phi)
    r33 = math.cos(theta) * math.cos(phi)

    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    return rotation_matrix

# reference: modern robotics, appendix B.1
def matrix2ZYXangles(rotation_matrix: np.ndarray) -> list:
    # ZYX_angle = [phi, theta, psi]
    # R = Rz(psi)Ry(theta)Rx(phi)

    assert type(rotation_matrix) == np.ndarray and rotation_matrix.shape == (3, 3)

    r11 = rotation_matrix[0][0]
    r12 = rotation_matrix[0][1]
    r21 = rotation_matrix[1][0]
    r22 = rotation_matrix[1][1]
    r31 = rotation_matrix[2][0]
    r32 = rotation_matrix[2][1]
    r33 = rotation_matrix[2][2]

    if r31 != 1 and r31 != -1:
        psi = math.atan2(r21, r11)
        theta = math.atan2(-r31, (r11 ** 2 + r21 ** 2) ** 0.5)
        phi = math.atan2(r32, r33)
    elif r31 == -1:
        psi = 0
        theta = math.pi / 2
        phi = math.atan2(r12, r22)
    else:
        psi = 0
        theta = -math.pi / 2
        phi = -math.atan2(r12, r22)

    ZYX_angles = [phi, theta, psi]
    return ZYX_angles

def matrix2quat(rotation_matrix: np.ndarray, quat_data_type: Optional[str] = 'list') -> Union[list, np.ndarray]:
    '''
    Reference: modern robotics, appendix B.3
    '''

    assert type(rotation_matrix) == np.ndarray and rotation_matrix.shape == (3, 3)
    assert quat_data_type == 'list' or quat_data_type == '1darray'

    r11 = rotation_matrix[0, 0]
    r22 = rotation_matrix[1, 1]
    r33 = rotation_matrix[2, 2]

    r12 = rotation_matrix[0, 1]
    r21 = rotation_matrix[1, 0]

    r13 = rotation_matrix[0, 2]
    r31 = rotation_matrix[2, 0]

    r23 = rotation_matrix[1, 2]
    r32 = rotation_matrix[2, 1]

    q0 = 0.5 * np.sqrt(1 + r11 + r22 + r33)

    coef = 1 / (4 * q0)
    q1 = coef * (r32 - r23)
    q2 = coef * (r13 - r31)
    q3 = coef * (r21 - r12)

    if quat_data_type == 'list':
        return [q0, q1, q2, q3]
    else:
        return np.array([q0, q1, q2, q3])

def vec2so3(vec: Union[list, np.ndarray]) -> np.ndarray:
    vec = vec_format_standardization(vec)
    assert len(vec) == 3

    so3 = np.zeros((3, 3))
    so3[1, 2] = -vec[0]
    so3[0, 2] = vec[1]
    so3[0, 1] = -vec[2]
    so3[2, 1] = vec[0]
    so3[2, 0] = -vec[1]
    so3[1, 0] = vec[2]
    return so3

def exp_so3(omega, theta) -> np.ndarray:
    omega = vec_format_standardization(omega)
    assert len(omega) == 3

    omega_ss = vec2so3(omega)
    exp_so3 = np.identity(3) + np.sin(theta) * omega_ss + \
        (1 - np.cos(theta)) * omega_ss @ omega_ss
    return exp_so3

def invT(T: np.ndarray) -> np.ndarray:
    assert type(T) == np.ndarray and T.shape == (4, 4)

    R = T[0:3, 0:3]
    p = T[0:3, 3]

    invT = np.zeros((4, 4))
    invT[0:3, 0:3] = R.T
    invT[0:3, 3] = -R.T @ p
    invT[3, 3] = 1
    return invT

def adSE3_T(SE3_matrix: np.ndarray) -> np.ndarray:
    assert type(SE3_matrix) == np.ndarray and SE3_matrix.shape == (4, 4)
    
    R = SE3_matrix[0:3, 0:3]
    p = SE3_matrix[0:3, 3]
    p_ss = vec2so3(p)

    adSE3 = np.zeros((6, 6))
    adSE3[0:3, 0:3] = R
    adSE3[3:6, 3:6] = R
    adSE3[3:6, 0:3] = p_ss @ R
    return adSE3

def adSE3_Rp(R: np.ndarray, p: Union[list, np.ndarray]) -> np.ndarray:
    p = vec_format_standardization(p)
    assert type(R) == np.ndarray and R.shape == (3, 3)
    assert len(p) == 3

    p_ss = vec2so3(p)

    adSE3 = np.zeros((6, 6))
    adSE3[0:3, 0:3] = R
    adSE3[3:6, 3:6] = R
    adSE3[3:6, 0:3] = p_ss @ R
    return adSE3

def Rp2T(R: np.ndarray, p: Union[list, np.ndarray]) -> np.ndarray:
    p = vec_format_standardization(p, std_type='1darray')
    assert type(p) == np.ndarray and p.shape == (3, )
    assert type(R) == np.ndarray and R.shape == (3, 3)

    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    T[3, 3] = 1
    return T

def exp_se3(screw_axes, theta) -> np.ndarray:
    S = vec_format_standardization(screw_axes)
    assert len(S) == 6
    
    omega = np.array([S[0], S[1], S[2]])
    v = np.array([S[3], S[4], S[5]])

    exp_se3 = np.zeros((4, 4))

    if np.linalg.norm(omega) == 1:
        omega_ss = vec2so3(omega)
        exp_omega = exp_so3(omega, theta)

        exp_se3[0:3, 0:3] = exp_omega
        exp_se3[0:3, 3] = (theta * np.identity(3) + (1 - np.cos(theta)) \
            * omega_ss + (theta - np.sin(theta)) * omega_ss @ omega_ss) @ v
        exp_se3[3, 3] = 1
    elif np.linalg.norm(omega) == 0 and np.linalg.norm(v) == 1:
        exp_se3[0:3, 0:3] = np.identity(3)
        exp_se3[0:3, 3] = theta * v
        exp_se3[3, 3] = 1
    else:
        print('ERROR: invalid se3.')
        return None

    return exp_se3

def compute_screw_axis(omega, q) -> np.ndarray:
    omega = vec_format_standardization(omega)
    q = vec_format_standardization(q)
    assert len(omega) == 3 and len(q) == 3

    omega = np.array(omega)
    q = np.array(q)
    v = -vec2so3(omega) @ q
    S = np.append(omega, v)
    return S


def twist2se3(twist) -> np.ndarray:
    twist = vec_format_standardization(twist)
    assert len(twist) == 6

    omega = twist[0:3]
    v = twist[3:6]

    se3 = np.zeros((4, 4))
    
    se3[0, 3] = v[0]
    se3[1, 3] = v[1]
    se3[2, 3] = v[2]

    omega_ss = vec2so3(omega)
    se3[0:3, 0:3] = omega_ss
    
    return se3
    
def fk_open_chain(home_config, screw_axes_list, theta_list) -> np.ndarray:
    assert len(screw_axes_list) == len(theta_list)

    num_joints = len(screw_axes_list)

    T = np.identity(4)
    for i in range(num_joints):
        Si = screw_axes_list[i]
        qi = theta_list[i]
        T = T @ exp_se3(Si, qi)
    T = T @ home_config

    return T

def main():
    quat = [0.7071, 0.7071, 0, 0]
    matrix = quat2matrix(quat)
    quat2 = matrix2quat(matrix)
    print(quat2)

if __name__ == '__main__':
    main()