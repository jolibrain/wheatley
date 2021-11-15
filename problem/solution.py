import numpy as np


class Solution:
    def __init__(self, schedule, real_durations):
        self.schedule = schedule
        self.real_durations = real_durations

    def get_makespan(self):
        return np.max(self.schedule + self.real_durations)
