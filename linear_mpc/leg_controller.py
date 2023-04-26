import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from robot_configs import RobotConfig
from robot_data import RobotData

class LegController():
    
    def __init__(self, robot_config: RobotConfig, is_initialization=False):
        self.is_initialization = is_initialization
        self.__load_parameters(robot_config)

    def __load_parameters(self, robot_config: RobotConfig):
        self.__Kp_swing = robot_config.Kp_swing
        self.__Kd_swing = robot_config.Kd_swing

    def update(self, f_mpc, robot_data: RobotData, cur_leg_states_info):
        self.__f_mpc = f_mpc
        self.__command_list = []
        self.__leg_states_info = cur_leg_states_info

        self.__compute_torque_command(robot_data)

    def __compute_torque_command(self, robot_data: RobotData):
        Jv_feet = robot_data.Jv_feet
        base_Jv_feet = robot_data.base_Jv_feet
        # vel_base = robot_data.vel_base
        R_base = robot_data.R_base
        base_vel_base_feet = robot_data.base_vel_base_feet
        base_pos_base_feet = robot_data.base_pos_base_feet
        # print(base_pos_base_feet)

        if self.is_initialization:
            for leg_id in range(4):
                Jvi = Jv_feet[leg_id]
                # tau_i = Jvi.T @ R_base.T @ -(self.__f_mpc[leg_id])
                tau_i = Jvi.T @ -(self.__f_mpc[leg_id])
                command_i = tau_i[6+3*leg_id : 6+3*(leg_id+1)]
                self.__command_list.append(command_i)
        else:
            for leg_id in range(4):
                # the case that leg is in stance state
                if self.__leg_states_info[leg_id] == 'stance':
                    Jvi = Jv_feet[leg_id]
                    # tau_i = Jvi.T @ R_base.T @ -(self.__f_mpc[leg_id])
                    tau_i = Jvi.T @ -(self.__f_mpc[leg_id])
                    command_i = tau_i[6+3*leg_id : 6+3*(leg_id+1)]
                    self.__command_list.append(command_i)
                # the case that leg is in swing state
                else:
                    Jvi = Jv_feet[leg_id]
                    [base_pos_swingfoot_des, base_vel_swingfoot_des] = self.__leg_states_info[leg_id]
                    base_vel_base_footi = base_vel_base_feet[leg_id]
                    base_pos_base_footi = base_pos_base_feet[leg_id]
                    # swing_err = self.__Kp_swing * (base_pos_swingfoot_des - base_pos_base_footi) \
                    #     + self.__Kd_swing * (base_vel_swingfoot_des - base_vel_base_footi)
                    swing_err = self.__Kp_swing @ (R_base @ base_pos_swingfoot_des - R_base @ base_pos_base_footi) \
                        + self.__Kd_swing @ (R_base @ base_vel_swingfoot_des - R_base @ base_vel_base_footi)
                    # print(swing_err)
                    # tau_i = base_Jvi.T @ swing_err
                    tau_i = Jvi.T @ swing_err
                    command_i = tau_i[6+3*leg_id : 6+3*(leg_id+1)]
                    self.__command_list.append(command_i)

    def get_torque_command(self):
        commands = []
        for i in range(4):
            for j in range(3):
                commands.append(self.__command_list[i][j])
        # print(self.__command_list)
        return commands