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

    def solve(self) -> np.ndarray:
        while np.any(self.schedule[:, :-1] == -1):
            current_time, machine_ids = self.priority_queue.get()
            failed_steps = [
                machine_id
                for machine_id in machine_ids
                if not self.step(machine_id, current_time)
            ]

            if 0 < len(failed_steps) < len(machine_ids):
                # Some tasks have been scheduled. We can try again.
                self.priority_queue.put((current_time, failed_steps))
            elif len(failed_steps) == len(machine_ids):
                # All tasks have failed, we can safely wait for more tasks to end.
                next_ending_time, next_machine_ids = self.priority_queue.get()
                next_machine_ids.extend(failed_steps)
                self.priority_queue.put((next_ending_time, next_machine_ids))

        validate_solution(
            self.durations,
            self.affectations,
            self.schedule[:, :-1],
        )
        return self.schedule[:, :-1]

    def step(self, machine_id: int, current_time: int) -> bool:
        """Update the priority queue and the current solution by adding
        a job for the given machine.
        """
        valid_candidates = self.candidates(machine_id, current_time)
        if len(valid_candidates) == 0:
            return False

        selected_job = self.priority_rule(valid_candidates)
        starting_time = self.canditate_starting_time(selected_job, current_time)

        task_id = self.schedule[selected_job].argmin()
        ending_time = starting_time + self.durations[selected_job, task_id]
        self.priority_queue.put((ending_time, [machine_id]))
        self.schedule[selected_job, task_id] = starting_time

        return True

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


def reschedule(
    durations: np.ndarray, affectations: np.ndarray, schedule: np.ndarray
) -> np.ndarray:
    """Adapt the current schedule to take into account the given durations.
    Does not modify the order of the schedule.

    ---
    Args:
        durations: The new durations of the tasks.
            Shape of [n_jobs, n_machines].
        affectations: The affectations of the tasks.
            Shape of [n_jobs, n_machines].
        schedule: The current schedule.
            Shape of [n_jobs, n_machines].

    ---
    Returns:
        The new schedule.
            Shape of [n_jobs, n_machines].
    """
    # Save the original tasks order of each machine.
    occupancy = _occupancy(affectations, schedule)
    # Start by a trivial schedule and make sure that each task job is at least
    # starting right after its precedency.
    new_schedule = _init_schedule(durations)

    # Iterate until we have a fixed point solution.
    # During each iteration, we separately fix the job constraints and the
    # machines constraints.
    while not np.all(schedule == new_schedule):
        schedule = new_schedule
        new_schedule = _reschedule_jobs(durations, new_schedule)
        new_schedule = _reschedule_machines(
            durations, affectations, occupancy, new_schedule
        )

    return new_schedule


def _reschedule_jobs(durations: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Modify the schedule to make sure that the job constraints
    are respected.
    """
    n_tasks = durations.shape[1]
    schedule = schedule.copy()

    for task_id in range(1, n_tasks):
        starting_times = np.stack(
            (
                schedule[:, task_id],
                schedule[:, task_id - 1] + durations[:, task_id - 1],
            ),
            axis=1,
        )
        schedule[:, task_id] = np.max(starting_times, axis=1)

    return schedule


def _reschedule_machines(
    durations: np.ndarray,
    affectations: np.ndarray,
    occupancy: np.ndarray,
    schedule: np.ndarray,
) -> np.ndarray:
    """Modify the schedule to make sure that the machine constraints
    are respected.
    """
    schedule = schedule.copy()

    # Order the tasks by the machine they are affected to.
    sort_by_machines = np.argsort(affectations, axis=1)
    schedule = np.take_along_axis(schedule, sort_by_machines, axis=1)
    durations = np.take_along_axis(durations, sort_by_machines, axis=1)

    # Read to [n_machines, n_jobs].
    schedule = schedule.transpose()
    durations = durations.transpose()

    # Order the tasks by the original occupancy schedule.
    schedule = np.take_along_axis(schedule, occupancy, axis=1)
    durations = np.take_along_axis(durations, occupancy, axis=1)

    # Make sure each task starts after the precedent task on each machine.
    n_tasks = durations.shape[1]
    for task_id in range(1, n_tasks):
        starting_times = np.stack(
            (
                schedule[:, task_id],
                schedule[:, task_id - 1] + durations[:, task_id - 1],
            ),
            axis=1,
        )
        schedule[:, task_id] = np.max(starting_times, axis=1)

    # Go back to the original schedule format.
    schedule = np.take_along_axis(schedule, np.argsort(occupancy), axis=1)
    schedule = schedule.transpose()
    schedule = np.take_along_axis(schedule, affectations, axis=1)
    return schedule


def _init_schedule(durations: np.ndarray) -> np.ndarray:
    """Initialize a schedule by starting each job task when its precedency
    is finished. This is important to make sure that no gap between two tasks
    is let unfilled.
    """
    schedule = np.zeros_like(durations)
    for task_id in range(1, durations.shape[1]):
        schedule[:, task_id] = schedule[:, task_id - 1] + durations[:, task_id - 1]

    return schedule


def _occupancy(affectations: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Compute the occupancy of each machine of the given schedule.
    The occupancy of a machine is the order a machine treat each job.

    ---
    Returns:
        The occupancy of each machine.
            Shape of [n_machines, n_jobs].
    """
    # Order the tasks by the machine they are affected to.
    sort_by_machines = np.argsort(affectations, axis=1)
    schedule = np.take_along_axis(schedule, sort_by_machines, axis=1)

    schedule = schedule.transpose()

    # Chronological order of the jobs for each machine.
    occupancy = np.argsort(schedule, axis=1)
    return occupancy
