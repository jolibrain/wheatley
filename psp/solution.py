#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
from itertools import accumulate


class Solution:
    @classmethod
    def from_mode_schedule(
        cls, mode_schedule, problem, affected, jobids, real_durations
    ):
        nmodes_per_job = [nj[0] for nj in problem["job_info"]]
        nmodes_per_job_cum = [0] + list(accumulate(nmodes_per_job))
        job_schedule = np.empty((problem["n_jobs"]), dtype=np.float32)
        modes = np.empty((problem["n_jobs"]), dtype=int)
        mode_offset = 0
        for m in range(mode_schedule.shape[0]):
            if affected[m]:
                job_schedule[jobids[m]] = mode_schedule[m]
                modes[jobids[m]] = m - nmodes_per_job_cum[jobids[m]]

        return cls(
            job_schedule=job_schedule,
            modes=modes,
            mode_schedule=mode_schedule,
            real_durations=real_durations,
        )

    def __init__(
        self, job_schedule=None, modes=None, mode_schedule=None, real_durations=None
    ):
        self.job_schedule = np.array(job_schedule)
        self.modes = np.array(modes)
        self.mode_schedule = np.array(mode_schedule)
        self.real_durations = np.array(real_durations)
        self.schedule = (self.job_schedule, self.modes)

    def get_makespan(self):
        return max(self.job_schedule + self.real_durations)
