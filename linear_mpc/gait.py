import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))

from enum import Enum
import numpy as np

from linear_mpc_configs import LinearMpcConfig

class Gait(Enum):
    '''
    name: name of the gait
    num_segment: 
    '''

    STANDING = 'standing', 16, np.array([0, 0, 0, 0]), np.array([16, 16, 16, 16])
    TROTTING16 = 'trotting', 16, np.array([0, 8, 8, 0]), np.array([8, 8, 8, 8])
    TROTTING10 = 'trotting', 10, np.array([0, 5, 5, 0]), np.array([5, 5, 5, 5])
    JUMPING16 = 'jumping', 16, np.array([0, 0, 0, 0]), np.array([4, 4, 4, 4])
    # BOUNDING8 = 'bounding', 8, np.array([4, 4, 0, 0]), np.array([4, 4, 4, 4])
    PACING16 = 'pacing', 16, np.array([8, 0, 8, 0]), np.array([8, 8, 8, 8])
    PACING10 = 'pacing', 10, np.array([5, 0, 5, 0]), np.array([5, 5, 5, 5])

    def __init__(
        self, 
        name: str, 
        num_segment: int, 
        stance_offsets: np.ndarray, 
        stance_durations: np.ndarray
    ) -> None:

        self.__name = name
        self.__num_segment = num_segment
        self.__stance_offsets = stance_offsets
        self.__stance_durations = stance_durations

        self.__load_parameters()

        # each leg has the same swing time and stance time in one period
        self.total_swing_time: int = num_segment - stance_durations[0]
        self.total_stance_time: int = stance_durations[0]

        # normalization
        self.stance_offsets_normalized = stance_offsets / num_segment
        self.stance_durations_normalized = stance_durations / num_segment

    def __load_parameters(self) -> None:
        self.__dt_control: float = LinearMpcConfig.dt_control
        self.__iterations_between_mpc: int = LinearMpcConfig.iteration_between_mpc
        self.__mpc_horizon: int = LinearMpcConfig.horizon

    @property
    def name(self) -> str:
        return self.__name

    @property
    def num_segment(self) -> int:
        return self.__num_segment

    @property
    def stance_offsets(self) -> np.ndarray:
        return self.__stance_offsets

    @property
    def stance_durations(self) -> np.ndarray:
        return self.__stance_durations

    @property
    def swing_time(self) -> float:
        return self.get_total_swing_time(self.__dt_control * self.__iterations_between_mpc)

    @property
    def stance_time(self) -> float:
        return self.get_total_stance_time(self.__dt_control * self.__iterations_between_mpc)

    def set_iteration(self, iterations_between_mpc: int, cur_iteration: int) -> None:
        self.iteration = np.floor(cur_iteration / iterations_between_mpc) % self.num_segment
        self.phase = (cur_iteration % (iterations_between_mpc * self.num_segment)) \
            / (iterations_between_mpc * self.num_segment)

    def get_gait_table(self) -> np.ndarray:
        '''
        compute gait table for force constraints in mpc

        1 for stance, 0 for swing
        '''
        gait_table = np.zeros(4 * self.__mpc_horizon, dtype=np.float32)
        for i in range(self.__mpc_horizon):
            i_horizon = (i + 1 + self.iteration) % self.num_segment
            cur_segment = i_horizon - self.stance_offsets
            for j in range(4):
                if cur_segment[j] < 0:
                    cur_segment[j] += self.num_segment
                
                if cur_segment[j] < self.stance_durations[j]:
                    gait_table[i*4+j] = 1
                else:
                    gait_table[i*4+j] = 0

        return gait_table

    def get_swing_state(self) -> np.ndarray:
        swing_offsets_normalizerd = self.stance_offsets_normalized + self.stance_durations_normalized
        for i in range(4):
            if(swing_offsets_normalizerd[i] > 1):
                swing_offsets_normalizerd -= 1
        swing_durations_normalized = 1 - self.stance_durations_normalized

        phase_state = np.array([self.phase, self.phase, self.phase, self.phase], dtype=np.float32)
        swing_state = phase_state - swing_offsets_normalizerd

        for i in range(4):
            if swing_state[i] < 0:
                swing_state[i] += 1
            
            if swing_state[i] > swing_durations_normalized[i]:
                swing_state[i] = 0
            else:
                swing_state[i] = swing_state[i] / swing_durations_normalized[i]

        return swing_state

    def get_stance_state(self) -> np.ndarray:
        phase_state = np.array([self.phase, self.phase, self.phase, self.phase], dtype=np.float32)
        stance_state = phase_state - self.stance_offsets_normalized
        for i in range(4):
            if stance_state[i] < 0:
                stance_state[i] += 1
            
            if stance_state[i] > self.stance_durations_normalized[i]:
                stance_state[i] = 0
            else:
                stance_state[i] = stance_state[i] / self.stance_durations_normalized[i]

        return stance_state

    def get_total_swing_time(self, dt_mpc: float) -> float:
        '''
        compute total swing time

        dt_mpc: dt between mpc (time between mpc) 
                i.e. dt_mpc = dt_control * iterations_between_mpc. 
        '''
        return dt_mpc * self.total_swing_time

    def get_total_stance_time(self, dt_mpc: float) -> float:
        '''
        compute total stance time
        '''
        return dt_mpc * self.total_stance_time


def main():
    standing = Gait.STANDING
    trotting = Gait.TROTTING10

    iteration_between_mpc = 20
    idx = 0

    print(trotting.swing_time)

    for i in range(1000):
        if i % iteration_between_mpc == 0:
            print('--------update MPC--------')
            print('idx =', idx)
            idx += 1
            trotting.set_iteration(iteration_between_mpc, i)
            print(trotting.get_gait_table())
            
        # print(trotting.get_cur_swing_time(30 * 0.001))
        # print(trotting.swing_time)

        # stance_state = trotting.get_stance_state()
        # swing_state = trotting.get_swing_state()
        # print('iteration: ', trotting.iteration)

        # print('phase: ', trotting.phase)
        # print('stance state: ', stance_state)
        # print('swing state: ', swing_state)



if __name__ == '__main__':
    main()

