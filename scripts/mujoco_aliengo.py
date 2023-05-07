import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

import mujoco_py
from mujoco_py import MjViewer
import numpy as np

from gait import Gait
from leg_controller import LegController
from linear_mpc_configs import LinearMpcConfig
from mpc import ModelPredictiveController
from robot_configs import AliengoConfig
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


STATE_ESTIMATION = False

def reset(sim):
    sim.reset()
    # q_pos_init = np.array([
    #     0, 0, 0.116536,
    #     1, 0, 0, 0,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77
    # ])
    q_pos_init = np.array([
        0, 0, 0.41,
        1, 0, 0, 0,
        0, 0.8, -1.6,
        0, 0.8, -1.6,
        0, 0.8, -1.6,
        0, 0.8, -1.6
    ])
    
    q_vel_init = np.array([
        0, 0, 0, 
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ])
    
    init_state = mujoco_py.cymj.MjSimState(
        time=0.0, 
        qpos=q_pos_init, 
        qvel=q_vel_init, 
        act=None, 
        udd_state={}
    )
    sim.set_state(init_state)

def get_true_simulation_data(sim):
    pos_base = sim.data.body_xpos[1]
    vel_base = sim.data.body_xvelp[1]
    quat_base = sim.data.sensordata[0:4]
    omega_base = sim.data.sensordata[4:7]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
    pos_foothold = [
        sim.data.get_geom_xpos('fl_foot'),
        sim.data.get_geom_xpos('fr_foot'),
        sim.data.get_geom_xpos('rl_foot'),
        sim.data.get_geom_xpos('rr_foot')
    ]
    vel_foothold = [
        sim.data.get_geom_xvelp('fl_foot'),
        sim.data.get_geom_xvelp('fr_foot'),
        sim.data.get_geom_xvelp('rl_foot'),
        sim.data.get_geom_xvelp('rr_foot')
    ]
    pos_thigh = [
        sim.data.get_body_xpos("FL_thigh"),
        sim.data.get_body_xpos("FR_thigh"),
        sim.data.get_body_xpos("RL_thigh"),
        sim.data.get_body_xpos("RR_thigh")
    ]

    true_simulation_data = [
        pos_base, 
        vel_base, 
        quat_base, 
        omega_base, 
        pos_joint, 
        vel_joint, 
        touch_state, 
        pos_foothold, 
        vel_foothold, 
        pos_thigh
    ]
    # print(true_simulation_data)
    return true_simulation_data

def get_simulated_sensor_data(sim):
    imu_quat = sim.data.sensordata[0:4]
    imu_gyro = sim.data.sensordata[4:7]
    imu_accelerometer = sim.data.sensordata[7:10]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
            
    simulated_sensor_data = [
        imu_quat, 
        imu_gyro, 
        imu_accelerometer, 
        pos_joint, 
        vel_joint, 
        touch_state
        ]
    # print(simulated_sensor_data)
    return simulated_sensor_data


def initialize_robot(sim, viewer, robot_config, robot_data):
    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)
    init_gait = Gait.STANDING
    vel_base_des = [0., 0., 0.]
    
    for iter_counter in range(800):

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

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
        contact_forces = predictive_controller.update_mpc_if_needed(
            iter_counter, vel_base_des, 0., gait_table, solver='drake', debug=False, iter_debug=0)

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states)
        sim.data.ctrl[:] = torque_cmds

        sim.step()
        viewer.render()

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
    # initialize_robot(sim, viewer, robot_config, robot_data)

    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)

    gait = Gait.TROTTING10
    swing_foot_trajs = [SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(4)]

    vel_base_des = np.array([1.4, 0., 0.])
    yaw_turn_rate_des = 0.

    iter_counter = 0

    while True:

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5]
        )

        gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
        swing_states = gait.get_swing_state()
        gait_table = gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)

        contact_forces = predictive_controller.update_mpc_if_needed(iter_counter, vel_base_des, 
            yaw_turn_rate_des, gait_table, solver='drake', debug=False, iter_debug=0) 

        pos_targets_swingfeet = np.zeros((4, 3))
        vel_targets_swingfeet = np.zeros((4, 3))

        for leg_idx in range(4):
            if swing_states[leg_idx] > 0:   # leg is in swing state
                swing_foot_trajs[leg_idx].set_foot_placement(
                    robot_data, gait, vel_base_des, yaw_turn_rate_des
                )
                base_pos_base_swingfoot_des, base_vel_base_swingfoot_des = \
                    swing_foot_trajs[leg_idx].compute_traj_swingfoot(
                        robot_data, gait
                    )
                pos_targets_swingfeet[leg_idx, :] = base_pos_base_swingfoot_des
                vel_targets_swingfeet[leg_idx, :] = base_vel_base_swingfoot_des

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states, pos_targets_swingfeet, vel_targets_swingfeet)
        sim.data.ctrl[:] = torque_cmds

        sim.step()
        viewer.render()
        iter_counter += 1


        if iter_counter == 50000:
            sim.reset()
            reset(sim)
            iter_counter = 0
            break

        
if __name__ == '__main__':
    main()