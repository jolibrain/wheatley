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
        cls,
        mode_schedule,
        problem,
        affected,
        jobids,
        real_durations,
        criterion,
        schedule_stoch=None,
    ):
        if isinstance(problem, dict):
            nmodes_per_job = [nj[0] for nj in problem["job_info"]]
        else:
            nmodes_per_job = problem.n_modes_per_job
        nmodes_per_job_cum = [0] + list(accumulate(nmodes_per_job))
        if isinstance(problem, dict):
            job_schedule = np.empty((problem["n_jobs"]), dtype=np.float32)
            modes = np.empty((problem["n_jobs"]), dtype=int)
        else:
            job_schedule = np.empty((problem.n_jobs), dtype=np.float32)
            modes = np.empty((problem.n_jobs), dtype=int)

        mode_offset = 0
        for m in range(mode_schedule.shape[0]):
            if affected[m]:
                job_schedule[jobids[m]] = mode_schedule[m]
                modes[jobids[m]] = m - nmodes_per_job_cum[jobids[m]]

        return cls(
            problem=problem,
            job_schedule=job_schedule,
            modes=modes,
            mode_schedule=mode_schedule,
            real_durations=real_durations,
            criterion=criterion,
            schedule_stoch=schedule_stoch,
        )

    def __init__(
        self,
        problem=None,
        job_schedule=None,
        modes=None,
        mode_schedule=None,
        real_durations=None,
        criterion=None,
        schedule_stoch=None,
    ):
        self.problem = problem
        self.job_schedule = np.array(job_schedule, dtype=np.float32)
        self.modes = np.array(modes)
        self.mode_schedule = np.array(mode_schedule)
        self.real_durations = np.array(real_durations, dtype=np.float32)
        self.schedule = (self.job_schedule, self.modes)
        self._criterion = criterion
        if schedule_stoch is not None:
            self.schedule_stoch = schedule_stoch.tolist()
        else:
            schedule_stoch = None

    def get_criterion(self):
        return self._criterion

    def save(self, path, label):
        with open(path, "w") as f:
            f.write(f"prolem: {label}\n")
            f.write(f"criterion value: {self._criterion}\n")
            f.write(f"njobs: {len(self.job_schedule)}\n")
            f.write("job_schedule starts (real)\n")
            for n, v in enumerate(self.job_schedule):
                f.write(f"{self.problem.job_labels[n]} : {v}\n")
            if self.schedule_stoch is not None:
                f.write("job_schedule starts  (mode, min, max)\n")
                for n, v in enumerate(self.schedule_stoch):
                    f.write(f"{self.problem.job_labels[n]} : {v}\n")
            f.write("durations (real):\n")
            for n, v in enumerate(self.real_durations):
                f.write(f"{self.problem.job_labels[n]} : {v}\n")
            if self.schedule_stoch is not None:
                f.write("durations  (mode)\n")
                for n, v in enumerate(self.problem.durations[0]):
                    f.write(f"{self.problem.job_labels[n]} : {v}\n")
                f.write("durations  (min)\n")
                for n, v in enumerate(self.problem.durations[1]):
                    f.write(f"{self.problem.job_labels[n]} : {v}\n")
                f.write("durations  (max)\n")
                for n, v in enumerate(self.problem.durations[2]):
                    f.write(f"{self.problem.job_labels[n]} : {v}\n")

        print("solution saved: ", path)
