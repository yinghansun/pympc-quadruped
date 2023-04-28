import os
import sys
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from robot_data import RobotData

class LegController():
    '''Compute torques for each joint.

    For stance legs, joint torques are computed with $ tau_i = Jv.T @ -f_i $, 
    where Jv (3 x 3 matrix) is the foot Jacobian, and f_i is the contact force 
    related to foot i, calculated by the predictive controller. All the 
    quantities are expressed in world frame.

    For swing legs, joint torques are computed by a PD controller (by 
    assuming the mass of the leg is not too large) to track the given swing 
    foot targets. All the quantities are expressed in world frame.

    Params
    ------
    Kp_swing: np.ndarray
        The P gain for swing leg PD control.
    Kd_swing: np.ndarray
        The D gain for swing leg PD control.
    '''
    def __init__(self, Kp_swing: np.ndarray, Kd_swing: np.ndarray):
        self.__Kp_swing = Kp_swing
        self.__Kd_swing = Kd_swing
        self.__torque_cmds = np.zeros(12, dtype=np.float16)

    @property
    def torque_cmds(self) -> np.ndarray:
        return self.__torque_cmds

    def update(
        self, 
        robot_data: RobotData, 
        contact_forces: np.ndarray, 
        swing_states: List[int],
        pos_targets_swingfeet: np.ndarray,
        vel_targets_swingfeet: np.ndarray
    ):
        '''Update joint torques using current data.

        Args
        ----
        robot_data: RobotData
            records current robot data. You need to update the current robot 
            data before computing the joint torques.
        contact_forces: np.ndarray, shape = (12, )
            contact forces of each foot expressed in world frame, computed 
            by the predictive controller.
        swing_states: List[int]
            identify whether each leg is in swing (=1) or stance (=0).
        pos_targets_swingfeet: np.ndarray, shape = (4, 3)
            target position of each swing foot relative to base, expressed in 
            base frame.
        vel_targets_swingfeet: np.ndarray, shape = (4, 3)
            target velocity of each swing foot relative to base, expressed in 
            base frame.

        Returns
        -------
        torque_cmds: np.ndarray, shape = (12, )
            torque commands of each joint.
        '''
        Jv_feet = robot_data.Jv_feet
        R_base = robot_data.R_base
        base_vel_base_feet = robot_data.base_vel_base_feet
        base_pos_base_feet = robot_data.base_pos_base_feet

        for leg_idx in range(4):
            Jvi = Jv_feet[leg_idx]

            if swing_states[leg_idx]:
                base_pos_swingfoot_des = pos_targets_swingfeet[leg_idx, :]
                base_vel_swingfoot_des = vel_targets_swingfeet[leg_idx, :]
                
                swing_err = self.__Kp_swing @ (R_base @ base_pos_swingfoot_des - R_base @ base_pos_base_feet[leg_idx]) \
                    + self.__Kd_swing @ (R_base @ base_vel_swingfoot_des - R_base @ base_vel_base_feet[leg_idx])                
                tau_i = Jvi.T @ swing_err
                cmd_i = tau_i[6+3*leg_idx : 6+3*(leg_idx+1)]
                self.__torque_cmds[3*leg_idx:3*(leg_idx+1)] = cmd_i
            else:
                tau_i = Jvi.T @ -contact_forces[3*leg_idx:3*(leg_idx+1)]
                cmd_i = tau_i[6+3*leg_idx:6+3*(leg_idx+1)]
                self.__torque_cmds[3*leg_idx:3*(leg_idx+1)] = cmd_i

        return self.__torque_cmds