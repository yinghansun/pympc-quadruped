import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import PiecewisePolynomial
from gait import Gait
from robot_data import RobotData

from linear_mpc_configs import LinearMpcConfig
from robot_configs import AliengoConfig

class SwingFootTrajectoryGenerator():

    def __init__(self, leg_id):
        self.__load_parameters()

        self.__is_first_swing = True
        self.__remaining_swing_time = 0.
        self.__leg_id = leg_id

        self.__footpos_init = np.zeros((3, 1), dtype=float)
        self.__footpos_final = np.zeros((3, 1), dtype=float)

    def __load_parameters(self):
        self.__dt_control = LinearMpcConfig.dt_control
        self.__swing_height = AliengoConfig.swing_height
        self.__gravity = LinearMpcConfig.gravity

    def __set_init_foot_position(self, footpos_init):
        self.__footpos_init = footpos_init

    def __set_final_foot_position(self, footpos_final):
        self.__footpos_final = footpos_final

    def generate_swing_foot_trajectory(self, total_swing_time, cur_swing_time):
        break_points = np.array([[0.],
                                 [total_swing_time / 2.0],
                                 [total_swing_time]], dtype=float)

        footpos_middle_time = (self.__footpos_init + self.__footpos_final) / 2
        footpos_middle_time[2] = self.__swing_height

        # print(footpos_middle_time)
        footpos_break_points = np.hstack((
            self.__footpos_init.reshape(3, 1),
            footpos_middle_time.reshape(3, 1),
            self.__footpos_final.reshape(3, 1)
        ))

        vel_break_points = np.zeros((3, 3), dtype=float)

        swing_traj = PiecewisePolynomial.CubicHermite(break_points, footpos_break_points, vel_break_points)

        # print(cur_swing_time)
        pos_swingfoot = swing_traj.value(cur_swing_time)
        vel_swingfoot = swing_traj.derivative(1).value(cur_swing_time)
        # acc_swingfoot = swing_traj.derivate(2).value(cur_swing_time)

        # x = np.linspace(start=0., stop=total_swing_time, num=100)
        # y = [swing_traj.value(xi)[0] for xi in x]
        # plt.plot(x, y)
        # plt.show()

        return np.squeeze(pos_swingfoot), np.squeeze(vel_swingfoot)
    
    def compute_swing_foot_trajectory_in_base_frame(self, robot_data: RobotData, gait: Gait):
        pos_base = np.array(robot_data.pos_base, dtype=float)
        vel_base = np.array(robot_data.lin_vel_base, dtype=float)
        R_base = robot_data.R_base

        total_swing_time = gait.swing_time
        cur_swing_time = total_swing_time - self.__remaining_swing_time
        pos_swingfoot_des, vel_swingfoot_des = self.generate_swing_foot_trajectory(total_swing_time, cur_swing_time)
        
        base_R_world = R_base.T
        base_pos_swingfoot_des = base_R_world @ (pos_swingfoot_des - pos_base)
        base_vel_swingfoot_des = base_R_world @ (vel_swingfoot_des - vel_base)

        return base_pos_swingfoot_des, base_vel_swingfoot_des

    def set_foot_init_and_final_placement(self, robot_data: RobotData, gait: Gait, base_vel_des, yaw_turn_rate_des):

        pos_base = np.array(robot_data.pos_base, dtype=float)
        vel_base = np.array(robot_data.lin_vel_base, dtype=float)
        R_base = robot_data.R_base
        base_pos_base_footi = robot_data.base_pos_base_feet[self.__leg_id]
        # print(robot_data.pos_feet[self.__leg_id])
        # print(base_pos_base_footi)
        base_pos_base_thighi = robot_data.base_pos_base_thighs[self.__leg_id]

        total_stance_time = gait.stance_time
        total_swing_time = gait.swing_time
        swing_state = gait.get_swing_state()[self.__leg_id]

        world_vel_des = R_base @ base_vel_des
        # world_vel_des = base_vel_des

        # update the remaining swing time
        if self.__is_first_swing:
            self.__remaining_swing_time = total_swing_time
        else:
            self.__remaining_swing_time -= self.__dt_control

        # foot placement
        RotZ = self.__get_RotZ(yaw_turn_rate_des * 0.5 * total_stance_time)
        pos_thigh_corrected = RotZ @ base_pos_base_thighi
        # print('position thigh', base_pos_base_thighi)
        # print('pos_thigh_corrected', pos_thigh_corrected)
        # print('remaining time', self.__remaining_swing_time)
        # print('base_vel_des', base_vel_des)

        world_footpos_final = pos_base + \
            R_base @ (pos_thigh_corrected + base_vel_des * self.__remaining_swing_time) + \
            0.5 * total_stance_time * vel_base + 0.03 * (vel_base - world_vel_des)
        # world_footpos_final = pos_base + \
        #     R_base @ (pos_thigh_corrected + base_vel_des * self.__remaining_swing_time)
        world_footpos_final[0] += (0.5 * pos_base[2] / self.__gravity) * (vel_base[1] * yaw_turn_rate_des)
        world_footpos_final[1] += (0.5 * pos_base[2] / self.__gravity) * (-vel_base[0] * yaw_turn_rate_des)
        world_footpos_final[2] = 0.024 # TODO: what's this?
        
        # world_footpos_final = robot_data.pos_feet[self.__leg_id]
        world_footpos_final[2] = 0.0        
        self.__set_final_foot_position(world_footpos_final)

        # TODO: This is different from drake project, is this right?
        if self.__is_first_swing:
            self.__is_first_swing = False
            # world_p_footi = pos_base + R_base @ base_pos_base_footi
            # self.__set_init_foot_position(world_p_footi)
            self.__set_init_foot_position(robot_data.pos_feet[self.__leg_id])
            # print(self.__footpos_init, self.__footpos_final)

        if swing_state >= 1:    # swing finished
            self.__is_first_swing = True

    def visualize_traj(self, x, y):
        plt.plot(x, y)
        plt.show()

    @staticmethod
    def __get_RotZ(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta),  0.],
                         [0.,            0.,             1.]])

def main():
    data = [[0.00727408, 0.00061764, 0.43571295],    # pos_base
            [0.0189759 , 0.00054278, 0.02322867],    # vel_base
            [9.99951619e-01, -9.13191258e-03,  3.57360542e-03,  7.72221709e-04],  # quat_base
            [-0.06964452, -0.01762341, -0.00088601], # omega_base
            [0.00687206, 0.52588717, -1.22975589, 
             0.02480081, 0.51914926, -1.21463939,
             0.00892169, 0.51229961, -1.20195572,
             0.02621839, 0.50635251, -1.18849609],   # pos_joint
            [0.06341452, -0.02158136, 0.16191205,
             0.07448259, -0.04855474, 0.21399941,
             0.06280346,  0.00562435, 0.10597827,
             0.07388069, -0.02180622, 0.15909948],   # vel_joint
            [43.75199071, 41.87769053, 55.08127523, 53.19869752],  # touch_state (fz)
            [[0.28034683, 0.13037568, 0.02508994],
             [0.28059001, -0.13029236,  0.02514821],
             [-0.19924435,  0.13046479,  0.02472057],
             [-0.19901342, -0.13039177,  0.02477902]], # pos_foothold
            [[0.00408774, -0.00251474, -0.00079255],
             [0.00480164,  0.00263784, -0.00071313],
             [0.00347752, -0.002389  , -0.00094539],
             [0.00420289,  0.00265997, -0.00086827]],  # vel_foothold
            [[0.24695605, 0.13495843, 0.43211885],
             [0.24736863, -0.13301709,  0.43438321],
             [-0.23282991,  0.13425052,  0.43572478],
             [-0.23241938, -0.13372591,  0.43770139]]] # pos_thigh

    robot_data = RobotData()
    robot_data.update(data)

    gait = Gait.TROTTING
    gait.set_iteration(30, 0)
    # swing_states = gait.get_swing_state()
    # # stance_states = gait.get_stance_state()
    # gait_table = gait.get_gait_table()

    test = SwingFootTrajectoryGenerator(0)
    test.set_foot_init_and_final_placement(robot_data, gait, np.array([0.5, 0., 0.]), 0.)
    base_pos_swingfoot_des, base_vel_swingfoot_des = test.compute_swing_foot_trajectory_in_base_frame(robot_data, gait)
    print(base_pos_swingfoot_des, base_vel_swingfoot_des)

    test1 = SwingFootTrajectoryGenerator(1)
    test1.set_foot_init_and_final_placement(robot_data, gait, np.array([0.5, 0., 0.]), 0.)
    base_pos_swingfoot_des, base_vel_swingfoot_des = test1.compute_swing_foot_trajectory_in_base_frame(robot_data, gait)
    print(base_pos_swingfoot_des, base_vel_swingfoot_des)

    test2 = SwingFootTrajectoryGenerator(2)
    test2.set_foot_init_and_final_placement(robot_data, gait, np.array([0.5, 0., 0.]), 0.)
    base_pos_swingfoot_des, base_vel_swingfoot_des = test2.compute_swing_foot_trajectory_in_base_frame(robot_data, gait)
    print(base_pos_swingfoot_des, base_vel_swingfoot_des)

    test3 = SwingFootTrajectoryGenerator(3)
    test3.set_foot_init_and_final_placement(robot_data, gait, np.array([0.5, 0., 0.]), 0.)
    base_pos_swingfoot_des, base_vel_swingfoot_des = test3.compute_swing_foot_trajectory_in_base_frame(robot_data, gait)
    print(base_pos_swingfoot_des, base_vel_swingfoot_des)

if __name__ == '__main__':
    main()