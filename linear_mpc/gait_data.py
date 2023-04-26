import numpy as np

class GaitData():
    def __init__(self):
        self.swing_time = 0
        self.stance_time = 0
        self.gait_table = np.array([])
        self.swing_states = np.zeros(4)
        self.contact_states = np.zeros(4)