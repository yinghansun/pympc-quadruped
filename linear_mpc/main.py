import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))

import mujoco_py
from mujoco_py import MjViewer
import matplotlib.pyplot as plt
import numpy as np

from gait import Gait
from leg_controller import LegController
from mpc import ModelPredictiveController
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator

from robot_configs import AliengoConfig
from linear_mpc_configs import LinearMpcConfig

STATE_ESTIMATION = False

def reset(sim):
    sim.reset()
    q_pos_init = np.array([0, 0, 0.116536,
                           1, 0, 0, 0,
                           0, 1.16, -2.77,
                           0, 1.16, -2.77,
                           0, 1.16, -2.77,
                           0, 1.16, -2.77])
    # q_pos_init = np.array([0, 0, 0.41,
    #                        1, 0, 0, 0,
    #                        0, 0.8, -1.6,
    #                        0, 0.8, -1.6,
    #                        0, 0.8, -1.6,
    #                        0, 0.8, -1.6])
    q_vel_init = np.array([0, 0, 0, 
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0])
    init_state = mujoco_py.cymj.MjSimState(time=0.0, qpos=q_pos_init, qvel=q_vel_init, act=None, udd_state={})
    sim.set_state(init_state)

def get_true_simulation_data(sim):
    pos_base = sim.data.body_xpos[1]
    vel_base = sim.data.body_xvelp[1]
    quat_base = sim.data.sensordata[0:4]
    omega_base = sim.data.sensordata[4:7]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
    pos_foothold = [sim.data.get_geom_xpos('fl_foot'),
                    sim.data.get_geom_xpos('fr_foot'),
                    sim.data.get_geom_xpos('rl_foot'),
                    sim.data.get_geom_xpos('rr_foot')]
    vel_foothold = [sim.data.get_geom_xvelp('fl_foot'),
                    sim.data.get_geom_xvelp('fr_foot'),
                    sim.data.get_geom_xvelp('rl_foot'),
                    sim.data.get_geom_xvelp('rr_foot')]
    pos_thigh = [sim.data.get_body_xpos("FL_thigh"),
                 sim.data.get_body_xpos("FR_thigh"),
                 sim.data.get_body_xpos("RL_thigh"),
                 sim.data.get_body_xpos("RR_thigh")]

    true_simulation_data = [pos_base, vel_base, quat_base, omega_base, pos_joint, 
                            vel_joint, touch_state, pos_foothold, vel_foothold, pos_thigh]
    # print(true_simulation_data)
    return true_simulation_data

def get_simulated_sensor_data(sim):
    imu_quat = sim.data.sensordata[0:4]
    imu_gyro = sim.data.sensordata[4:7]
    imu_accelerometer = sim.data.sensordata[7:10]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
            
    simulated_sensor_data = [imu_quat, imu_gyro, imu_accelerometer, pos_joint, vel_joint, touch_state]
    # print(simulated_sensor_data)
    return simulated_sensor_data

def convert_force_vector_to_matrix(vec):
    f_mat = [vec[0:3], vec[3:6], vec[6:9], vec[9:12]]
    return f_mat

def get_leg_states_info():
    pass

def initialize_robot(sim, viewer, robot_config, robot_data):
    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig)
    leg_controller = LegController(robot_config, is_initialization=True)
    init_gait = Gait.STANDING
    vel_base_des = [0., 0., 0.]
    
    for iter_counter in range(800):

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)
        # robot_data.update(data)
        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5]
        )

        init_gait.set_iteration(predictive_controller.iterations_between_mpc,iter_counter)
        swing_states = init_gait.get_swing_state()
        gait_table = init_gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)
        f_mpc = predictive_controller.update_mpc_if_needed(
            iter_counter, vel_base_des, 0., gait_table, solver='drake', debug=False, iter_debug=0)[0:12]
        f_mpc = convert_force_vector_to_matrix(f_mpc)

        leg_controller.update(f_mpc, robot_data, swing_states)
        torque_command = leg_controller.get_torque_command()
        sim.data.ctrl[:] = torque_command

        sim.step()
        viewer.render()

def visualize_planner_result(pos_base_list, base_pos_base_swingfoot_des_list):
    x = range(0, len(pos_base_list))
    plt.plot(x, pos_base_list, label='base x-position')
    plt.plot(x, base_pos_base_swingfoot_des_list[0], label='fl-foot x-pos des')
    plt.plot(x, base_pos_base_swingfoot_des_list[1], label='fr-foot x-pos des')
    plt.plot(x, base_pos_base_swingfoot_des_list[2], label='rl-foot x-pos des')
    plt.plot(x, base_pos_base_swingfoot_des_list[3], label='rr-foot x-pos des')
    plt.legend()
    plt.show()

def main():
    cur_path = os.path.dirname(__file__)
    mujoco_xml_path = os.path.join(cur_path, '../robot/aliengo/aliengo.xml')
    model = mujoco_py.load_model_from_path(mujoco_xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = MjViewer(sim)

    reset(sim)
    sim.step()

    robot_config = AliengoConfig

    urdf_path = os.path.join(cur_path, '../robot/aliengo/urdf/aliengo.urdf')
    robot_data = RobotData(urdf_path, state_estimation=STATE_ESTIMATION)
    initialize_robot(sim, viewer, robot_config, robot_data)

    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig)
    leg_controller = LegController(robot_config)

    gait = Gait.TROTTING10
    swing_foot_trajectories = [SwingFootTrajectoryGenerator(leg_id) for leg_id in range(4)]

    vel_base_des = np.array([1.5, 0., 0.])
    yaw_turn_rate_des = 0.

    iter_counter = 0

    xpos_base_list = []
    xworld_pos_world_swingfoot_des_list = [[], [], [], []]

    while True:

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

        # robot_data.update(data)
        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5]
        )

        xpos_base_list.append(robot_data.pos_base[0])

        gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
        swing_states = gait.get_swing_state()
        gait_table = gait.get_gait_table()
        # print(gait_table)

        predictive_controller.update_robot_state(robot_data)

        f_mpc = predictive_controller.update_mpc_if_needed(iter_counter, vel_base_des, 
            yaw_turn_rate_des, gait_table, solver='drake', debug=False, iter_debug=0) 
        f_mpc = convert_force_vector_to_matrix(f_mpc)
        # print(f_mpc)

        '''
        initialize a dictionary. the key of this dictionary denotes the leg id,
        the value are the information about each leg. if the leg is in stance, then
        the corresponding value should be 'stance', else if the leg is in swing,
        the corresponding value shoule be [base_pos_base_swingfoot_des, 
        base_vel_base_swingfoot_des]
        '''
        leg_states_info = {0: None, 1: None, 2: None, 3: None}
        for leg_id in range(4):
            if swing_states[leg_id] > 0:   # leg is in swing state
                swing_foot_trajectories[leg_id].set_foot_init_and_final_placement(
                    robot_data, gait, vel_base_des, yaw_turn_rate_des)
                base_pos_base_swingfoot_des, base_vel_base_swingfoot_des = \
                    swing_foot_trajectories[leg_id].compute_swing_foot_trajectory_in_base_frame(
                    robot_data, gait)
                # print(base_pos_base_swingfoot_des, base_vel_base_swingfoot_des)
                leg_states_info[leg_id] = [base_pos_base_swingfoot_des, base_vel_base_swingfoot_des]

                pos_swingfoot_des = robot_data.pos_base + robot_data.R_base @ base_pos_base_swingfoot_des
                xworld_pos_world_swingfoot_des_list[leg_id].append(pos_swingfoot_des[0])
            else:
                leg_states_info[leg_id] = 'stance'
                xworld_pos_world_swingfoot_des_list[leg_id].append(robot_data.pos_feet[leg_id][0])

        leg_controller.update(f_mpc, robot_data, leg_states_info)
        torque_command = leg_controller.get_torque_command()
        sim.data.ctrl[:] = torque_command

        sim.step()
        viewer.render()
        iter_counter += 1


        if iter_counter == 50000:
            sim.reset()
            reset(sim)
            iter_counter = 0
            # visualize_planner_result(xpos_base_list, xworld_pos_world_swingfoot_des_list)
            break

        
if __name__ == '__main__':
    main()