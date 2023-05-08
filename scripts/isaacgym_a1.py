import os
import sys
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

from isaacgym import gymapi, gymtorch
import numpy as np

from gait import Gait
from leg_controller import LegController
from linear_mpc_configs import LinearMpcConfig
from isaacgym_utils import *
from mpc import ModelPredictiveController
from robot_configs import A1Config, AliengoConfig
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


NUM_ROBOTS = 4
STATE_ESTIMATION = False

def reset(gym, sim, robot_config):
    base_state_tensor_discription = gym.acquire_actor_root_state_tensor(sim)
    base_state_tensor = gymtorch.wrap_tensor(base_state_tensor_discription)
    base_state_tensor[:, 0:3] = torch.zeros_like(base_state_tensor[:, 0:3])  # base position
    base_state_tensor[:, 2] = robot_config.base_height_des
    base_state_tensor[:, 3:7] = torch.zeros_like(base_state_tensor[:, 3:7])  # base orientation
    base_state_tensor[:, 6] = 1.
    base_state_tensor[:, 7:10] = torch.zeros_like(base_state_tensor[:, 7:10]) # base linear velocity
    base_state_tensor[:, 10:13] = torch.zeros_like(base_state_tensor[:, 10:13]) # base angular velocity
    base_state_tensor_discription = gymtorch.unwrap_tensor(base_state_tensor)
    gym.set_actor_root_state_tensor(sim, base_state_tensor_discription)
    
    # joint_state_tensor_discription = gym.acquire_dof_state_tensor(sim)
    # joint_state_tensor = gymtorch.wrap_tensor(joint_state_tensor_discription)
    # joint_state_tensor[:, 1] = torch.zeros_like(joint_state_tensor[:, 1])
    # for robot_idx in range(NUM_ROBOTS):
    #     joint_state_tensor[0*robot_idx, 0] = 0
    #     joint_state_tensor[1*robot_idx, 0] = 0.8
    #     joint_state_tensor[2*robot_idx, 0] = -1.6
    #     joint_state_tensor[3*robot_idx, 0] = 0
    #     joint_state_tensor[4*robot_idx, 0] = 0.8
    #     joint_state_tensor[5*robot_idx, 0] = -1.6
    #     joint_state_tensor[6*robot_idx, 0] = 0
    #     joint_state_tensor[7*robot_idx, 0] = 0.8
    #     joint_state_tensor[8*robot_idx, 0] = -1.6
    #     joint_state_tensor[9*robot_idx, 0] = 0
    #     joint_state_tensor[10*robot_idx, 0] = 0.8
    #     joint_state_tensor[11*robot_idx, 0] = -1.6
    #     joint_state_tensor_discription = gymtorch.unwrap_tensor(joint_state_tensor)
    # gym.set_dof_state_tensor(sim, joint_state_tensor_discription)



def main():
    gym = gymapi.acquire_gym()
    sim = create_simulation(
        gym, 
        dt=LinearMpcConfig.dt_control, 
        use_gpu_pipeline=True
    )

    robot_config = A1Config

    asset_root_path = os.path.join(os.path.dirname(__file__), '../robot/')
    robot_path = 'a1/urdf/a1.urdf'
    urdf_path = os.path.join(asset_root_path, robot_path)
    
    l_robot_model = []
    l_robot_data: List[RobotData] = []
    for _ in range(NUM_ROBOTS):
        l_robot_model.append(load_model(gym, sim, asset_root_path, robot_path, is_base_fixed=False))
        l_robot_data.append(RobotData(urdf_path, state_estimation=STATE_ESTIMATION))

    creat_ground_plane(gym, sim)

    for robot_model in l_robot_model:
        envs, actors = create_envs_actors(gym, sim, robot_model, robot_config.base_height_des)

    viewer = add_viewer(gym, sim)

    gym.prepare_sim(sim)

    reset(gym, sim, robot_config)

    l_predictive_controller: List[ModelPredictiveController] = []
    l_leg_controller: List[LegController] = []
    l_gait: List[Gait] = []
    l_swing_foot_trajs: List[SwingFootTrajectoryGenerator] = []
    for _ in range(NUM_ROBOTS):
        l_predictive_controller.append(ModelPredictiveController(LinearMpcConfig, A1Config))
        l_leg_controller.append(LegController(robot_config.Kp_swing, robot_config.Kd_swing))
        l_gait.append(Gait.TROTTING10)
        l_swing_foot_trajs.append([SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(4)])

    vel_base_des = np.array([1.4, 0., 0.])
    yaw_turn_rate_des = 0.

    iter_counter = 0
    render_counter = 0
    torque_cmds_tensor = torch.zeros(NUM_ROBOTS, 12, device='cuda:0', dtype=torch.float32)

    while not gym.query_viewer_has_closed(viewer):
        # step graphics
        if render_counter % 30 == 0:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            render_counter = 0
        render_counter += 1

        base_state_tensor_discription = gym.acquire_actor_root_state_tensor(sim)
        base_state_tensor: torch.Tensor = gymtorch.wrap_tensor(base_state_tensor_discription)
        joint_state_tensor_discription = gym.acquire_dof_state_tensor(sim)
        joint_state_tensor: torch.Tensor = gymtorch.wrap_tensor(joint_state_tensor_discription)

        for robot_idx, (robot_data, predictive_controller, leg_controller, gait, swing_foot_trajs) \
            in enumerate(zip(l_robot_data, l_predictive_controller, l_leg_controller, l_gait, l_swing_foot_trajs)):
            quat_base_imre=base_state_tensor[robot_idx, 3:7].cpu().numpy()
            quat_base_reim = np.array(
                [quat_base_imre[3], quat_base_imre[0], quat_base_imre[1], quat_base_imre[2]],
                dtype=np.float32
            )
            robot_data.update(
                pos_base=base_state_tensor[robot_idx, 0:3].cpu().numpy(),
                quat_base=quat_base_reim,
                lin_vel_base=base_state_tensor[robot_idx, 7:10].cpu().numpy(),
                ang_vel_base=base_state_tensor[robot_idx, 10:13].cpu().numpy(),
                q = joint_state_tensor[12*robot_idx:12*(robot_idx+1), 0].cpu().numpy(),
                qdot = joint_state_tensor[12*robot_idx:12*(robot_idx+1), 1].cpu().numpy()
            )
            print(robot_data.pos_base)
            print(robot_data.q)

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
                torque_cmds_tensor[robot_idx, :] = torch.tensor(torque_cmds, dtype=torch.float32, device='cuda:0')
            torque_cmds_tensor_discription = gymtorch.unwrap_tensor(torque_cmds_tensor)
            gym.set_dof_actuation_force_tensor(sim, torque_cmds_tensor_discription)

        gym.fetch_results(sim, True)
        gym.simulate(sim)

        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        iter_counter += 1

        if iter_counter == 3000:
            reset(gym, sim, robot_config)
            iter_counter = 0
            render_counter = 0

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == '__main__':
    main()