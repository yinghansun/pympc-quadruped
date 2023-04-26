import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pinocchio

from kinematics import adSE3_Rp, quat2matrix, quat2ZYXangle, vec_format_standardization


class RobotData():
    '''
    This framework is to use the sensor data (measurement) to generate all the 
    data of the robot in our control strategy.

    Params
    ------
    robot_model: str
        path to urdf file of the robot.
    state_estimation: (optional) bool
        identify whether to use the state estimation module. If the state estimation 
        module is used, we add noises to the data to simulate the real case. 
        Otherwise we use the data provided by the simulator directly.

    Measurement
    -----------
    1. the position $q$ and velocity $\dot{q}$ of each joint, provided by the 
    incremental encoder.
    2. the position $p$, velocity $v$, orientation $R$ and angular velocity 
    $\omega$ of the base, provided by IMU. Note that the orientation is expressed 
    as quanternion.

    Sign Convention
    ---------------
    1. quanternion is recorded as (real part, imaginary part).
    2. If we want to express a quantity B with respect to C in frame A, we write 
    A_quantity_C_B. By default, if the quantity does not have the prefix A, it 
    means this quantity is expressed in world frame, or inertia frame. In such 
    case, the prefix A is omitted. Similarly, if the quantity does not have the 
    suffix C, it means this quantity is relative to the world, or the inertia 
    frame. Now we show some examples:
    Ex.1. base_pos_base_foot: the position of foot with respect to base expressed 
    in base frame.
    Ex.2. pos_base_foot: the position of foot with respect to base expressed in 
    world frame. (it is the same as world_pos_base_foot)
    Ex.3. pos_foot: the position of foot with respect world expressed in world 
    frame. (it is the same as world_pos_world_foot)
    '''

    def __init__(
        self,
        robot_model: str, 
        state_estimation: Optional[bool] = False
    ) -> None:
        self.state_estimation = state_estimation
        self.__init_pinocchio(robot_model)

        self.__contact_history = np.zeros((4, 3), dtype=float)
    
    def update(
        self, 
        # data: Union[list, np.ndarray],
        pos_base: Union[list, np.ndarray],
        lin_vel_base: Union[list, np.ndarray],
        quat_base: Union[list, np.ndarray],
        ang_vel_base: Union[list, np.ndarray],
        q: Union[list, np.ndarray],
        qdot: Union[list, np.ndarray]
    ) -> None:
        if not self.state_estimation:
            self.pos_base: np.ndarray = vec_format_standardization(pos_base, '1darray')
            self.lin_vel_base: np.ndarray = vec_format_standardization(lin_vel_base, '1darray')
            self.quat_base: np.ndarray = vec_format_standardization(quat_base, '1darray')
            self.ang_vel_base: np.ndarray = vec_format_standardization(ang_vel_base, '1darray')

            self.R_base = quat2matrix(self.quat_base)
            self.rpy_base = np.array(quat2ZYXangle(self.quat_base))

            self.q: np.ndarray = vec_format_standardization(q, '1darray')
            self.qdot: np.ndarray = vec_format_standardization(qdot, '1darray')
        else:
            raise NotImplementedError

        # NOTE: quat in mujoco / Isaac Gym: (real part, imaginary part)
        #       quat in pinocchio: (imaginary part, real part)
        quat_base_converted = [self.quat_base[1], self.quat_base[2], 
            self.quat_base[3], self.quat_base[0]]
        # generalized joint position (floating base), dim: 3 + 3 + 12 = 18.
        generalized_q = list(self.pos_base) + list(quat_base_converted) \
            + list(self.q)
        self.__generalized_q = np.array(generalized_q, dtype=float)
        pinocchio.forwardKinematics(self.__pin_model, self.__pin_data, self.__generalized_q)
        pinocchio.framesForwardKinematics(self.__pin_model, self.__pin_data, self.__generalized_q)

        # NOTE: jacobian in pinocchio is [Jv, Jw], thus this X is actually X.T defined in modern robotics
        self.X_base = adSE3_Rp(self.R_base, self.pos_base)

        self.Jv_feet, self.base_Jv_feet = self.__compute_foot_Jacobian()   # geometric jacobian
        # the position of feet relative to world expressed in world frame
        self.pos_feet = self.__compute_pos_feet()
        # the position of feet relative to base expressed in world frame
        self.pos_base_feet = self.__compute_pos_base_feet()
        # the position of feet relative to base expressed in base frame
        self.base_pos_base_feet = self.__compute_base_pos_base_feet()
        self.base_vel_base_feet = self.__compute_base_vel_base_feet()
        # print(self.base_vel_base_feet)

        self.pos_thighs = self.__compute_pos_thighs()
        self.base_pos_base_thighs = self.__compute_base_pos_base_thighs()

    def __init_pinocchio(self, robot_model) -> None:
        # NOTE: The second parameter represents the floating base.
        # see https://github.com/stack-of-tasks/pinocchio/issues/1596
        self.__pin_model = pinocchio.buildModelFromUrdf(
            robot_model, pinocchio.JointModelFreeFlyer())
        self.__pin_data = self.__pin_model.createData()
        
    def __compute_foot_Jacobian(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        foot_frames = ['FL_foot_fixed', 'FR_foot_fixed', 'RL_foot_fixed', 'RR_foot_fixed']
        foot_frames_ID = [self.__pin_model.getFrameId(foot_frames[i]) for i in range(4)]
        
        Jv_feet = []
        base_Jv_feet = []
        for i in range(4):
            Ji = pinocchio.computeFrameJacobian(self.__pin_model, self.__pin_data, 
                self.__generalized_q, frame_id=foot_frames_ID[i], 
                reference_frame=pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # Jacobian in Pinocchio: [[Jv]
            #                         [Jw]]
            base_Ji = self.X_base @ Ji
            Jv_feet.append(Ji[0:3, :])
            base_Jv_feet.append(base_Ji[0:3, :])
            
        return Jv_feet, base_Jv_feet

    def __compute_pos_feet(self) -> List[np.ndarray]:
        foot_frames = ['FL_foot_fixed', 'FR_foot_fixed', 'RL_foot_fixed', 'RR_foot_fixed']
        foot_frames_ID = [self.__pin_model.getFrameId(foot_frames[i]) for i in range(4)]
        pos_feet = []
        for foot_idx in range(4):
            pos_footi = self.__pin_data.oMf[foot_frames_ID[foot_idx]].translation
            pos_feet.append(pos_footi)
        return pos_feet

    def __compute_pos_base_feet(self) -> List[np.ndarray]:
        pos_base_feet = []
        for foot_idx in range(4):
            pos_base_footi = self.pos_feet[foot_idx] - self.pos_base
            pos_base_feet.append(pos_base_footi)
        return pos_base_feet

    def __compute_base_pos_base_feet(self) -> List[np.ndarray]:
        base_pos_base_feet = []
        for foot_idx in range(4):
            base_pos_base_footi = self.R_base.T @ self.pos_base_feet[foot_idx]
            base_pos_base_feet.append(base_pos_base_footi)
        return base_pos_base_feet

    def __compute_base_vel_base_feet(self) -> List[np.ndarray]:
        base_vel_base_feet = []
        for foot_idx in range(4):
            generalized_qdot = list(self.lin_vel_base) + list(self.ang_vel_base) + list(self.qdot)
            generalized_qdot = np.array(generalized_qdot, dtype=float)
            vel_footi = self.Jv_feet[foot_idx] @ generalized_qdot
            vel_base_footi = vel_footi - self.lin_vel_base
            base_vel_base_footi = self.R_base.T @ vel_base_footi
            base_vel_base_feet.append(base_vel_base_footi)
        return base_vel_base_feet

    def __compute_pos_thighs(self) -> List[np.ndarray]:
        thigh_frames = ['FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint']
        thigh_frames_ID = [self.__pin_model.getFrameId(thigh_frames[i]) for i in range(4)]
        pos_thighs = []
        for thigh_idx in range(4):
            pos_thighi = self.__pin_data.oMf[thigh_frames_ID[thigh_idx]].translation
            pos_thighs.append(pos_thighi)
        return pos_thighs
        
    def __compute_base_pos_base_thighs(self) -> List[np.ndarray]:
        base_pos_base_thighs = []
        for thigh_idx in range(4):
            pos_base_thighi = self.pos_thighs[thigh_idx] - self.pos_base
            base_pos_base_thighi = self.R_base.T @ pos_base_thighi
            base_pos_base_thighs.append(base_pos_base_thighi)
        return base_pos_base_thighs
    
    def init_contact_history(self) -> None:
        self.__contact_history == self.pos_feet

    def __update_contact_history(self, cur_contact_state: Union[list, np.ndarray]) -> None:
        for foot_idx in range(4):
            if cur_contact_state[foot_idx] == 1:
                self.__contact_history[foot_idx] = self.pos_feet[foot_idx]

    def update_terrain_normal(self, cur_contact_state: Union[list, np.ndarray]) -> None:
        self.__update_contact_history(cur_contact_state)

        # Least squares approach
        #      A        x  =   b
        # [1 p1x p1y] [a0]   [p1z]
        # [1 p2x p2y] [a1] = [p2z]
        # [1 p3x p3y] [a2]   [p3z]
        # [1 p4x p4y]        [p4z]
        # A = np.array([
        #     [1, self.__contact_history[0, 0], self.__contact_history[0, 1]],
        #     [1, self.__contact_history[1, 0], self.__contact_history[1, 1]],
        #     [1, self.__contact_history[2, 0], self.__contact_history[2, 1]],
        #     [1, self.__contact_history[3, 0], self.__contact_history[3, 1]]
        # ])
        # b = np.array([
        #     [self.__contact_history[0, 2]],
        #     [self.__contact_history[1, 2]],
        #     [self.__contact_history[2, 2]],
        #     [self.__contact_history[3, 2]]
        # ])

        # x = np.linalg.lstsq(A, b)[0]

        # PCA approach
        X = self.__contact_history.T
        mu = np.mean(self.__contact_history, axis=0).reshape(3, 1)
        one = np.ones((1, 4), dtype=float)
        sigma = (X - mu @ one) @ ((X - mu @ one).T)
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        self.terrain_normal_est = eigenvectors[np.argmin(eigenvalues)]
        if self.terrain_normal_est[2] < 0:
            self.terrain_normal_est = -self.terrain_normal_est
        self.terrain_normal_est_yaw_aligned = self.R_base.T @ self.terrain_normal_est
        # print(self.terrain_normal_est_yaw_aligned)

def test():
    robot_path = os.path.join(os.path.dirname(__file__), '../robot/aliengo/urdf/aliengo.urdf')

    robot_data = RobotData(robot_model=robot_path)
    robot_data.update(
        pos_base=[0.00727408, 0.00061764, 0.43571295],
        lin_vel_base=[0.0189759 , 0.00054278, 0.02322867],
        quat_base=[9.99951619e-01, -9.13191258e-03,  3.57360542e-03,  7.72221709e-04],
        ang_vel_base=[-0.06964452, -0.01762341, -0.00088601],
        q=[0.00687206, 0.52588717, -1.22975589, 
           0.02480081, 0.51914926, -1.21463939,
           0.00892169, 0.51229961, -1.20195572,
           0.02621839, 0.50635251, -1.18849609],
        qdot=[0.06341452, -0.02158136, 0.16191205,
              0.07448259, -0.04855474, 0.21399941,
              0.06280346,  0.00562435, 0.10597827,
              0.07388069, -0.02180622, 0.15909948],
    )

    print(robot_data.base_pos_base_feet)

if __name__ == '__main__':
    test()