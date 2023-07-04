from queue import PriorityQueue

import numpy as np
from einops import repeat


class Solver:
    """Solver using a simulation rollout with dispatching rules.

    ---
    Args:
        processing_times: Processing times of the jobs.
            Shape of [n_jobs, n_machines].
        machines: Machine specification of the jobs.
            Shape of [n_jobs, n_machines].
    """

    def __init__(self, processing_times: np.ndarray, machines: np.ndarray):
        n_jobs, n_machines = processing_times.shape
        assert (
            processing_times.shape == machines.shape
        ), "Wrong number of jobs or machines"
        assert (
            machines.min() == 0 and machines.max() == n_machines - 1
        ), "The indices must be in the range [0, n_machines - 1]"
        ordered_index = np.arange(n_machines)
        assert np.all(
            np.sort(machines, axis=1) == repeat(ordered_index, "m -> n m", n=n_jobs)
        ), "The machines are not all used once for each job"

        self.processing_times = processing_times
        self.machines = machines
        self.n_jobs, self.n_machines = processing_times.shape

        # -1 for unknown starting times.
        self.starting_times = (
            np.ones((self.n_jobs, self.n_machines + 1), dtype=np.int32) * -1
        )
        self.priority_queue = PriorityQueue(maxsize=self.n_machines)

        # The first events are empty.
        for machine_id in range(self.n_machines):
            self.priority_queue.put((0, machine_id))

    def solve(self) -> int:
        while np.any(self.starting_times[:, :-1] == -1):
            current_time, machine_id = self.priority_queue.get()
            self.step(machine_id, current_time)

        ending_times = self.starting_times[:, :-1] + self.processing_times
        makespan = ending_times.max()
        return makespan

    def step(self, machine_id: int, current_time: int):
        """Update the priority queue and the current solution by adding
        a job for the given machine.
        """
        valid_candidates = self.candidates(machine_id)
        if len(valid_candidates) == 0:
            next_ending_time, next_machine_id = self.priority_queue.get()
            self.priority_queue.put((next_ending_time, next_machine_id))
            self.priority_queue.put((next_ending_time + 1, machine_id))
            return

        selected_job = self.priority_rule(valid_candidates)
        starting_time = self.canditate_starting_time(selected_job, current_time)

        task_id = self.starting_times[selected_job].argmin()
        ending_time = starting_time + self.processing_times[selected_job, task_id]
        self.priority_queue.put((ending_time, machine_id))
        self.starting_times[selected_job, task_id] = starting_time

    def candidates(self, machine_id: int) -> np.ndarray:
        """Select the valid candidates.
        A candidate is valid if it is the next unplaced task in its job,
        and if that task is to be done on the given `machine_id`.

        The returned candidates can be an empty array if there is no
        valid candidate.

        ---
        Args:
            machine_id: The machine onto which we filter the valid candidates.

        ---
        Returns:
            The indices of the jobs for which we consider their next task as
            valid candidates.
                Shape of [n_valid_candidates,]
        """
        job_ids = np.arange(self.n_jobs)
        machine_ids = np.arange(self.n_machines)
        machine_ids = repeat(machine_ids, "m -> n m", n=self.n_jobs)

        # If a job is fully done, its frontier candidate will have an id of `n_machines`.
        # This candidate will thus be ignored as the `machine_candidates` cannot be equal
        # to `n_machines`.
        frontier_candidates = self.starting_times.argmin(axis=1)  # Shape of [n_jobs,]

        # Since each machine is only present once per job,
        # `machine_candidates` is of shape [n_jobs,].
        machine_candidates = machine_ids[self.machines == machine_id]

        valid_jobs = job_ids[machine_candidates == frontier_candidates]
        return valid_jobs

    def priority_rule(self, candidates: np.ndarray) -> int:
        """Choose a candidate among the selected ones."""
        rng = np.random.default_rng(0)
        return rng.choice(candidates)

    def canditate_starting_time(self, job_id: int, current_time: int) -> int:
        """Determine the candidate starting time, which is either
        the current time or the time it takes for its previous task to finish.
        """
        task_id = self.starting_times[job_id].argmin()

        if task_id == 0:
            return current_time

        previous_starting_time = self.starting_times[job_id, task_id - 1]
        previous_process_time = self.processing_times[job_id, task_id - 1]
        previous_ending_time = previous_starting_time + previous_process_time

        return max(previous_ending_time, current_time)
