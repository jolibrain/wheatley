from typing import Optional

import numpy as np

from instances.generate_taillard import generate_taillard

from .rcpsp import Rcpsp


class TaillardRcpsp(Rcpsp):
    """Instantiate a randomly generated taillard instance and modelise the instance
    as a RCPSP problem.
    """

    def __init__(
        self,
        pb_id: int,
        n_jssp_jobs: int,
        n_jssp_machines: int,
        seed: Optional[int] = None,
    ):
        self.n_jssp_jobs = n_jssp_jobs
        self.n_jssp_machines = n_jssp_machines
        self.seed = seed

        self.durations, self.affectations = generate_taillard(
            n_jssp_jobs, n_jssp_machines, seed
        )

        # In RCPSP, the number of jobs is the total number of tasks in JSSP.
        # It also count the source and sink nodes.
        n_jobs = n_jssp_jobs * n_jssp_machines + 2
        source_id, sink_id = 1, n_jobs

        n_modes_per_job = [1 for _ in range(n_jobs)]
        task_ids = np.arange(n_jobs - 2) + 2  # The first task has the id 2.
        task_ids = task_ids.reshape(n_jssp_jobs, n_jssp_machines)

        # Successors.
        successors = np.roll(task_ids, shift=-1, axis=1)
        successors[:, -1] = sink_id
        # successors = list(successors.reshape(-1, 1))
        successors = [[s] for s in successors.reshape(-1)]
        source_successors = list(task_ids[:, 0])
        sink_successors = []
        successors = [source_successors] + successors + [sink_successors]

        # Durations.
        durations = list(self.durations.reshape(-1))
        # Add source and sink durations.
        durations = [0] + durations + [0]
        # Add the mode dimension.
        durations = [[d] for d in durations]
        # Add the dmin and dmax duration bounds.
        # durations = [[d, d, d] for d in durations]
        durations = [durations, durations, durations]

        # Resources constraints.
        affectations = self.affectations.reshape(-1)
        resources_cons = [
            [
                1 if r == affectations[task_id] else 0
                for r in range(1, n_jssp_machines + 1)
            ]
            for task_id in range(n_jobs - 2)
        ]
        resources_cons = (
            [[0 for _ in range(n_jssp_machines)]]  # Source.
            + resources_cons
            + [[0 for _ in range(n_jssp_machines)]]  # Sink.
        )
        # Add the mode dimension.
        resources_cons = [[r] for r in resources_cons]

        resources_availabilities = [1 for _ in range(n_jssp_machines)]
        n_renewable_resources = n_jssp_machines
        n_nonrenewable_resources = 0
        n_doubly_constrained_resources = 0

        super().__init__(
            pb_id,
            n_jobs,
            n_modes_per_job,
            successors,
            durations,
            resources_cons,
            resources_availabilities,
            n_renewable_resources,
            n_nonrenewable_resources,
            n_doubly_constrained_resources,
        )

    def sample(self, sampling_type: Optional[str] = None) -> "TaillardRcpsp":
        """Sample a new taillard instance."""
        seed = self.seed + 1 if self.seed is not None else None
        return TaillardRcpsp(self.pb_id, self.n_jssp_jobs, self.n_jssp_machines, seed)
