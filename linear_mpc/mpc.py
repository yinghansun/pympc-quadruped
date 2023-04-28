import os
import sys
import time
from typing import Union

sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import matplotlib.pyplot as plt
from numba import jit, vectorize, float32
import numpy as np
from scipy.linalg import expm
from pydrake.all import MathematicalProgram, Solve
from qpsolvers import solve_qp

from dynamics import make_com_inertial_matrix
from kinematics import quat2ZYXangle, vec2so3
from linear_mpc_configs import LinearMpcConfig
from robot_configs import RobotConfig
from robot_data import RobotData


class ModelPredictiveController():

    def __init__(self, mpc_config: LinearMpcConfig, robot_config: RobotConfig):        
        # state: x(t) = [\theta, p, \omega, \dot{p}, g], dim: 13
        self.num_state = 3 + 3 + 3 + 3 + 1
        # input: u(t) = [f1, f2, f3, f4], dim: 12
        self.num_input = 3 + 3 + 3 + 3 
        
        self.is_initialized = False
        self.is_first_run = True

        self._load_parameters(mpc_config, robot_config)
    
    def _load_parameters(self, mpc_config: LinearMpcConfig, robot_config: RobotConfig):        
        self.dt_control = mpc_config.dt_control
        self.iterations_between_mpc = mpc_config.iteration_between_mpc
        self.dt = 0.05
        self.horizon = mpc_config.horizon
        self.mu = mpc_config.friction_coef
        self.fz_max = robot_config.fz_max
        self.gravity = mpc_config.gravity

        self.body_I = make_com_inertial_matrix(
                ixx=0.033260231, ixy=-0.000451628, ixz=0.000487603, iyy=0.16117211, iyz=4.8356e-05, izz=0.17460442
            )
        self.mass = robot_config.mass_base
        self.com_height_des = robot_config.base_height_des

        # MPC weights
        _Qi = mpc_config.Q
        self.Qbar = np.kron(np.identity(self.horizon), _Qi)
        _Ri = mpc_config.R
        self.Rbar = np.kron(np.identity(self.horizon), _Ri)

    # data: [pos_base, vel_base, quat_base, omega_base, pos_joint, vel_joint, touch_state, pos_foothold, pos_thigh]
    def update_robot_state(self, robot_data: RobotData):
        if self.is_initialized == False:
            self.current_state = np.zeros(13, dtype=np.float32)
            self.roll_init = 0.0
            self.pitch_init = 0.0
            self.is_initialized = True

        self.__robot_data = robot_data

        # update xt
        rpy_base = quat2ZYXangle(robot_data.quat_base)
        pos_base = np.array(robot_data.pos_base, dtype=np.float32)
        # print(pos_base[2])
        omega_base = np.array(robot_data.ang_vel_base, dtype=np.float32)
        vel_base = np.array(robot_data.lin_vel_base, dtype=np.float32)
        for i in range(3):
            self.current_state[i] = rpy_base[i]
            self.current_state[3+i] = pos_base[i]
            self.current_state[6+i] = omega_base[i]
            self.current_state[9+i] = vel_base[i]
        # NOTE: this should be negative!!!
        self.current_state[12] = -self.gravity
        self.yaw = rpy_base[2]
        # update ri
        self.pos_base_feet = robot_data.pos_base_feet

    def update_mpc_if_needed(self, iter_counter, base_vel_base_des, yaw_turn_rate_des, 
        gait_table, solver='drake', debug=False, iter_debug=None):
        vel_base_des = self.__robot_data.R_base @ base_vel_base_des
        if self.is_first_run:
            self.xpos_base_desired = 0.0
            self.ypos_base_desired = 0.0
            self.yaw_desired = self.yaw
            self.is_first_run = False
        else:
            self.xpos_base_desired += self.dt_control * vel_base_des[0]
            self.ypos_base_desired += self.dt_control * vel_base_des[1]
            self.yaw_desired = self.yaw + self.dt_control * yaw_turn_rate_des
        
        # decide whether MPC needs to be updated
        if iter_counter % self.iterations_between_mpc == 0:
            ref_traj = self.generate_reference_trajectory(vel_base_des, yaw_turn_rate_des)
            self.ref_traj = ref_traj
            solve_start = time.time()
            self.__contact_forces = self._solve_mpc(ref_traj, gait_table, solver=solver)[0:12]
            solve_end = time.time()
            print('MPC solved in {:3f}s.'.format(solve_end - solve_start))
            print(self.yaw, self.yaw_desired)

            if debug and iter_counter == iter_debug:
                contact_forces_debug = self._solve_mpc(ref_traj, gait_table, solver=solver, debug=debug)
                self.__visulize_com_traj_solution(contact_forces_debug)

        return self.__contact_forces[0:12]

    def generate_reference_trajectory(
        self, 
        vel_base_des: Union[list, np.ndarray], 
        yaw_turn_rate: float
    ) -> np.ndarray:
        
        # vel_base_des = self.__robot_data.R_base @ base_vel_base_des

        cur_xpos_desired = self.xpos_base_desired
        cur_ypos_desired = self.ypos_base_desired
            
        max_pos_error = 0.1   # define the threshold for error of position
            
        '''
        compare the desired robot position and the current robot position.
        if the error is beyond this threshold, the current position (plus
        the threshold) is used as the desired position. else the desired 
        position does not change.
        '''
        if cur_xpos_desired - self.current_state[3] > max_pos_error:
            cur_xpos_desired = self.current_state[3] + max_pos_error
        if self.current_state[3] - cur_xpos_desired > max_pos_error:
            cur_xpos_desired = self.current_state[3] - max_pos_error

        if cur_ypos_desired - self.current_state[4] > max_pos_error:
            cur_ypos_desired = self.current_state[4] + max_pos_error
        if self.current_state[4] - cur_ypos_desired > max_pos_error:
            cur_ypos_desired = self.current_state[4] - max_pos_error

        self.xpos_base_desired = cur_xpos_desired
        self.ypos_base_desired = cur_ypos_desired

        # pitch and roll compensation
        if np.fabs(self.current_state[9]) > 0.2:
            self.pitch_init += self.dt * (0.0 - self.current_state[1]) / self.current_state[9]
        if np.fabs(self.current_state[10]) > 0.1:
            self.roll_init += self.dt * (0.0 - self.current_state[0]) / self.current_state[10]

        # staturation for pitch and roll compensation
        self.roll_init = np.fmin(np.fmax(self.roll_init, -0.25), 0.25)
        self.pitch_init = np.fmin(np.fmax(self.pitch_init, -0.25), 0.25)
        roll_comp = self.current_state[10] * self.roll_init
        pitch_comp = self.current_state[9] * self.pitch_init

        X_ref = np.zeros(self.num_state * self.horizon, dtype=np.float32)        
        X_ref[0::self.num_state] = roll_comp
        X_ref[1::self.num_state] = pitch_comp
        X_ref[2] = self.yaw_desired
        X_ref[3] = cur_xpos_desired
        X_ref[4] = cur_ypos_desired
        X_ref[5::self.num_state] = self.com_height_des
        X_ref[8::self.num_state] = yaw_turn_rate
        X_ref[9::self.num_state] = vel_base_des[0]
        X_ref[10::self.num_state] = vel_base_des[1]
        X_ref[12::self.num_state] = -self.gravity
        for i in range(1, self.horizon):
            X_ref[2 + self.num_state*i] = X_ref[2 + self.num_state*(i-1)] + self.dt * yaw_turn_rate
            X_ref[3 + self.num_state*i] = X_ref[3 + self.num_state*(i-1)] + self.dt * vel_base_des[0]
            X_ref[4 + self.num_state*i] = X_ref[4 + self.num_state*(i-1)] + self.dt * vel_base_des[1]

        return X_ref

    # dynamic constraints: \dot{x} = A_{c}x + B_{c}u
    def _generate_state_space_model(self):
        # Ac (13 * 13), Bc (13 * 12)
        Ac = np.zeros((self.num_state, self.num_state), dtype=np.float32)
        Bc = np.zeros((self.num_state, self.num_input), dtype=np.float32)

        Rz = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                       [np.sin(self.yaw), np.cos(self.yaw), 0],
                       [0, 0, 1]], dtype=np.float32)
        # Rz = self.__robot_data.R_base
        world_I = Rz @ self.body_I @ Rz.T
        
        Ac[0:3, 6:9] = Rz.T
        Ac[3:6, 9:12] = np.identity(3, dtype=np.float32)
        Ac[11, 12] = 1.0

        for i in range(4):
            Bc[6:9, 3*i:3*i+3] = np.linalg.inv(world_I) @ vec2so3(self.pos_base_feet[i])
            Bc[9:12, 3*i:3*i+3] = np.identity(3, dtype=np.float32) / self.mass
        
        return Ac, Bc

    def _discretize_continuous_model(self, Ac, Bc):
        # square_matrix = [[Ac (13*13), Bc (13*12)],
        #                  [0  (12*13), 0  (12*12)]] * dt (25*25)
        dim = self.num_state + self.num_input
        square_matrix = np.zeros((dim, dim), dtype=np.float32)
        square_matrix[0:self.num_state, 0:self.num_state] = Ac * self.dt
        square_matrix[0:self.num_state, self.num_state:dim] = Bc * self.dt

        #     [[Ac, Bc],          [[Ad, Bd],
        # exp( [ 0,  0]] * dt) =   [ 0,  I]]
        matrix_exponential = expm(square_matrix)
        Ad = matrix_exponential[0:self.num_state, 0:self.num_state]
        Bd = matrix_exponential[0:self.num_state, self.num_state:dim]

        return Ad, Bd

    # @jit(nopython=False)
    def _generate_QP_cost(self, Ad, Bd, xt, Xref, debug=False):
        # power_of_A = [A, A^2, A^3, ..., A^k]
        power_of_A = [np.identity(self.num_state, dtype=np.float32)]
        for i in range(self.horizon):
            power_of_A.append(power_of_A[i] @ Ad)

        Sx = np.zeros((self.num_state * self.horizon, self.num_state), dtype=np.float32)
        Su = np.zeros((self.num_state * self.horizon, self.num_input * self.horizon), dtype=np.float32)
        
        if debug:
            self.Sx = Sx
            self.Su = Su
            self.xt = xt

        for i in range(self.horizon):
            Sx[self.num_state*i:self.num_state*(i+1), 0:self.num_state] = power_of_A[i+1]

            for j in range(self.horizon):
                if i >= j:
                    Su[self.num_state*i:self.num_state*(i+1), self.num_input*j:self.num_input*(j+1)] = power_of_A[i-j] @ Bd

        qp_H = 2 * (Su.T @ self.Qbar @ Su + self.Rbar)
        qp_g = 2 * Su.T @ self.Qbar @ (Sx @ xt - Xref)

        return qp_H, qp_g
    
    def _generate_QP_constraints(self, gait_table):
        # friction cone constraint for one foot
        constraint_coef_matrix = np.array([
            [ 1,  0, self.mu],
            [-1,  0, self.mu],
            [ 0,  1, self.mu],
            [ 0, -1, self.mu],
            [ 0,  0,       1]
        ], dtype=np.float32)
        qp_C = np.kron(np.identity(4 * self.horizon, dtype=np.float32), constraint_coef_matrix)
        
        C_lb = np.zeros(4 * 5 * self.horizon, dtype=np.float32)
        C_ub = np.zeros(4 * 5 * self.horizon, dtype=np.float32)
        k = 0
        for i in range(self.horizon):
            for j in range(4):    # number of legs
                C_ub[5*k] = np.inf
                C_ub[5*k+1] = np.inf
                C_ub[5*k+2] = np.inf            
                C_ub[5*k+3] = np.inf
                C_ub[5*k+4] = gait_table[4*i+j] * self.fz_max
                k += 1

        return qp_C, C_lb, C_ub

    def _solve_mpc(self, ref_traj, gait_table, solver='drake', debug=False):

        assert solver == 'drake' or solver == 'qpsolvers'

        Ac, Bc = self._generate_state_space_model()
        self._discretize_continuous_model(Ac, Bc)
        Ad, Bd = self._discretize_continuous_model(Ac, Bc)
        # start_time = time.time()
        qpH, qpg = self._generate_QP_cost(Ad, Bd, self.current_state, ref_traj, debug=debug)
        # end_time = time.time()
        # print('QP cost generated in {:3f}s.'.format(end_time - start_time))

        # constraint_coef_matrix, lb, ub = self._generate_force_constraints(gait_table)
        qp_C, C_lb, C_ub = self._generate_QP_constraints(gait_table)

        if solver == 'drake':
            qp_problem = MathematicalProgram()
            contact_forces = qp_problem.NewContinuousVariables(self.num_input*self.horizon, 'contact_forces')

            qp_problem.AddQuadraticCost(qpH, qpg, contact_forces)
            qp_problem.AddLinearConstraint(qp_C, C_lb, C_ub, contact_forces)

            result = Solve(qp_problem)

            return result.GetSolution(contact_forces)

        else:
            contact_forces = solve_qp(P=qpH, q=qpg, G=qp_C, h=C_ub, A=None, b=None, solver='osqp')
            return contact_forces


    def __visulize_com_traj_solution(self, contact_forces_debug):
        com_traj = self.Sx @ self.xt + self.Su @ contact_forces_debug
        
        trajs = []
        for i in range(12):
            trajs.append([])
        
        for i in range(self.horizon):
            for j in range(12):
                trajs[j].append(com_traj[13*i+j])

        x = range(self.horizon)
        # plt.plot(x, trajs[0], label='roll')
        # plt.plot(x, trajs[1], label='pitch')
        # plt.plot(x, trajs[2], label='yaw')
        plt.plot(x, trajs[3], label='x')
        plt.plot(x, trajs[4], label='y')
        plt.plot(x, trajs[5], label='z')
        # plt.plot(x, trajs[6], label='wx')
        # plt.plot(x, trajs[7], label='wy')
        # plt.plot(x, trajs[8], label='wz')
        # plt.plot(x, trajs[9], label='vx')
        # plt.plot(x, trajs[10], label='vy')
        # plt.plot(x, trajs[11], label='vz')
        plt.legend()
        plt.show()
