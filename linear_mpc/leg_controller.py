import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from robot_configs import RobotConfig
from robot_data import RobotData

class LegController():
    
    def __init__(self, robot_config: RobotConfig):
        self.__load_parameters(robot_config)

    def __load_parameters(self, robot_config: RobotConfig):
        self.__Kp_swing = robot_config.Kp_swing
        self.__Kd_swing = robot_config.Kd_swing

    def update(
        self, 
        contact_forces: np.ndarray, 
        robot_data: RobotData, 
        swing_states: list,
        pos_targets_swingfeet: np.ndarray,
        vel_targets_swingfeet: np.ndarray
    ):
        self.__contact_forces = contact_forces
        self.__command_list = []
        self.__swing_states = swing_states
        self.__pos_targets_swingfeet = pos_targets_swingfeet
        self.__vel_targets_swingfeet = vel_targets_swingfeet

        self.__compute_torque_command(robot_data)

    def __compute_torque_command(self, robot_data: RobotData):
        Jv_feet = robot_data.Jv_feet
        R_base = robot_data.R_base
        base_vel_base_feet = robot_data.base_vel_base_feet
        base_pos_base_feet = robot_data.base_pos_base_feet

        for leg_idx in range(4):
            Jvi = Jv_feet[leg_idx]

            if self.__swing_states[leg_idx]:
                base_pos_swingfoot_des = self.__pos_targets_swingfeet[leg_idx, :]
                base_vel_swingfoot_des = self.__vel_targets_swingfeet[leg_idx, :]
                base_vel_base_footi = base_vel_base_feet[leg_idx]
                base_pos_base_footi = base_pos_base_feet[leg_idx]
                
                swing_err = self.__Kp_swing @ (R_base @ base_pos_swingfoot_des - R_base @ base_pos_base_footi) \
                    + self.__Kd_swing @ (R_base @ base_vel_swingfoot_des - R_base @ base_vel_base_footi)                
                tau_i = Jvi.T @ swing_err
                cmd_i = tau_i[6+3*leg_idx : 6+3*(leg_idx+1)]
                self.__command_list.append(cmd_i)
            else:
                tau_i = Jvi.T @ -self.__contact_forces[3*leg_idx:3*(leg_idx+1)]
                cmd_i = tau_i[6+3*leg_idx:6+3*(leg_idx+1)]
                self.__command_list.append(cmd_i)

    def get_torque_command(self):
        commands = []
        for i in range(4):
            for j in range(3):
                commands.append(self.__command_list[i][j])
        # print(self.__command_list)
        return commands