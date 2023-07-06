from queue import PriorityQueue

import numpy as np
from einops import repeat

from .heuristics import HEURISTICS
from .validate import validate_instance, validate_solution


class Solver:
    """Solver using a simulation rollout with dispatching rules.

    ---
    Args:
        processing_times: Processing times of the jobs.
            Shape of [n_jobs, n_machines].
        machines: Machine specification of the jobs.
            Shape of [n_jobs, n_machines].
    """

    def __init__(
        self,
        durations: np.ndarray,
        affectations: np.ndarray,
        heuristic: str,
        ignore_unfinished_precedences: bool,
    ):
        n_jobs, n_machines = durations.shape
        validate_instance(durations, affectations)
        assert heuristic in HEURISTICS, f"Unknown heuristic {heuristic}"

        self.durations = durations
        self.affectations = affectations
        self.n_jobs, self.n_machines = durations.shape
        self.ignore_unfinished_precedences = ignore_unfinished_precedences
        self.heuristic = HEURISTICS[heuristic]

        # -1 for unknown starting times.
        self.schedule = np.ones((self.n_jobs, self.n_machines + 1), dtype=np.int32) * -1
        self.priority_queue = PriorityQueue(maxsize=self.n_machines)

        # The first events are empty.
        self.priority_queue.put((0, list(range(self.n_machines))))

    def solve(self) -> int:
        while np.any(self.schedule[:, :-1] == -1):
            current_time, machine_ids = self.priority_queue.get()
            for machine_id in machine_ids:
                self.step(machine_id, current_time)

        validate_solution(
            self.durations,
            self.affectations,
            self.schedule[:, :-1],
        )
        return self.schedule[:, :-1]

    def step(self, machine_id: int, current_time: int):
        """Update the priority queue and the current solution by adding
        a job for the given machine.
        """
        valid_candidates = self.candidates(machine_id, current_time)
        if len(valid_candidates) == 0:
            if self.priority_queue.qsize() != 0:
                next_ending_time, next_machine_ids = self.priority_queue.get()
                next_machine_ids.append(machine_id)
            else:
                next_ending_time = current_time
                next_machine_ids = [machine_id]

            self.priority_queue.put((next_ending_time, next_machine_ids))
            return

        selected_job = self.priority_rule(valid_candidates)
        starting_time = self.canditate_starting_time(selected_job, current_time)

        task_id = self.schedule[selected_job].argmin()
        ending_time = starting_time + self.durations[selected_job, task_id]
        self.priority_queue.put((ending_time, [machine_id]))
        self.schedule[selected_job, task_id] = starting_time

    def candidates(self, machine_id: int, current_time: int) -> np.ndarray:
        """Select the valid candidates.
        A candidate is valid if it is the next unplaced task in its job,
        and if that task is to be done on the given `machine_id`.

        The returned candidates can be an empty array if there is no
        valid candidate.

        ---
        Args:
            machine_id: The machine onto which we filter the valid candidates.
            current_time: The current time of the simulation.

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
        frontier_candidates = self.schedule.argmin(axis=1)  # Shape of [n_jobs,]

        # Since each machine is only present once per job,
        # `machine_candidates` is of shape [n_jobs,].
        machine_candidates = machine_ids[self.affectations == machine_id]

        # Ignore frontier candidates that do not concern the given `machine_id`.
        filter = machine_candidates == frontier_candidates

        if self.ignore_unfinished_precedences:
            # Find the ending time of each precedent frontier candidate.
            # In case of a starting candidate, its precedent ending time will be 0.
            ending_times = self.schedule[:, :-1] + self.durations
            ending_times = np.concatenate(
                (np.zeros((self.n_jobs, 1), dtype=np.int32), ending_times),
                axis=1,
            )
            precedences_indices = np.expand_dims(frontier_candidates, axis=1)
            precedences_ending_times = np.take_along_axis(
                ending_times, precedences_indices, axis=1
            )
            precedences_ending_times = np.squeeze(precedences_ending_times, axis=1)

            # Also ignore tasks that have unfinished precedences.
            filter = filter & (precedences_ending_times <= current_time)

        valid_jobs = job_ids[filter]
        return valid_jobs

    def priority_rule(self, candidates: np.ndarray) -> int:
        """Choose a candidate among the selected ones."""
        return self.heuristic(
            self.durations,
            self.affectations,
            self.schedule,
            candidates,
        )

    def canditate_starting_time(self, job_id: int, current_time: int) -> int:
        """Determine the candidate starting time, which is either
        the current time or the time it takes for its previous task to finish.
        """
        task_id = self.schedule[job_id].argmin()

        if task_id == 0:
            return current_time

        previous_starting_time = self.schedule[job_id, task_id - 1]
        previous_process_time = self.durations[job_id, task_id - 1]
        previous_ending_time = previous_starting_time + previous_process_time

        return max(previous_ending_time, current_time)
