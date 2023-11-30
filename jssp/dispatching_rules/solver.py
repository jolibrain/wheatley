from queue import PriorityQueue

import numpy as np
from einops import repeat

from .heuristics import HEURISTICS
from .validate import (
    validate_instance,
    validate_instance_with_fictive_tasks,
    validate_solution,
    validate_solution_with_missing_tasks,
)


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
        n_machines = affectations.shape[1]
        if np.any(affectations == n_machines):
            validate_instance_with_fictive_tasks(durations, affectations)
        else:
            validate_instance(durations, affectations)

        assert heuristic in HEURISTICS, f"Unknown heuristic {heuristic}"

        self.durations = durations
        self.affectations = affectations
        self.n_jobs, self.n_machines = durations.shape
        self.fictive_machine_id = self.n_machines
        self.ignore_unfinished_precedences = ignore_unfinished_precedences
        self.heuristic = HEURISTICS[heuristic]
        self.durations_type = self.durations.dtype

        # -1 for unknown starting times.
        self.schedule = (
            np.ones((self.n_jobs, self.n_machines + 1), dtype=self.durations_type) * -1
        )
        self.priority_queue = PriorityQueue(maxsize=0)

        # The first events are empty.
        self.priority_queue.put((0, np.unique(self.affectations)))

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

        if np.any(self.affectations == self.n_machines):
            validate_solution_with_missing_tasks(
                self.durations,
                self.affectations,
                self.schedule[:, :-1],
            )
        else:
            validate_solution(
                self.durations,
                self.affectations,
                self.schedule[:, :-1],
            )
        return self.schedule[:, :-1]

    def step(self, machine_id: int, current_time: float) -> bool:
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

    def candidates(self, machine_id: int, current_time: float) -> np.ndarray:
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

        # If a job is fully done, its frontier candidate will have an id of `n_machines`.
        frontier_candidates = self.schedule.argmin(axis=1)  # Shape of [n_jobs,].

        affectations = np.concatenate(
            (self.affectations, np.zeros((self.n_jobs, 1), dtype=int)), axis=1
        )
        affectations[:, -1] = -1
        candidates_machine_id = affectations[job_ids, frontier_candidates]

        # Ignore frontier candidates that do not concern the given `machine_id`.
        valid_mask = candidates_machine_id == machine_id

        if self.ignore_unfinished_precedences:
            # Find the ending time of each precedent frontier candidate.
            # In case of a starting candidate, its precedent ending time will be 0.
            ending_times = self.schedule[:, :-1] + self.durations
            ending_times = np.concatenate(
                (np.zeros((self.n_jobs, 1), dtype=self.durations_type), ending_times),
                axis=1,
            )
            precedences_ending_times = ending_times[job_ids, frontier_candidates]

            # Also ignore tasks that have unfinished precedences.
            # Take into account the limited precision of floating-point comparisons.
            unfinished_precedences = (precedences_ending_times - current_time) < 1e-5
            valid_mask = valid_mask & unfinished_precedences

        valid_jobs = job_ids[valid_mask]
        return valid_jobs

    def priority_rule(self, candidates: np.ndarray) -> int:
        """Choose a candidate among the selected ones."""
        return self.heuristic(
            self.durations,
            self.affectations,
            self.schedule,
            candidates,
        )

    def canditate_starting_time(self, job_id: int, current_time: float) -> float:
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
    while not np.allclose(schedule, new_schedule):
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
    occupancy: list,
    schedule: np.ndarray,
) -> np.ndarray:
    """Modify the schedule to make sure that the machine constraints
    are respected.
    """
    n_machines = affectations.shape[1]
    schedule = schedule.copy()

    for machine_id in range(n_machines):
        if not np.any(affectations == machine_id):
            continue

        schedule_machine = schedule[affectations == machine_id]
        durations_machine = durations[affectations == machine_id]
        occupancy_machine = occupancy[machine_id]

        schedule_machine = schedule_machine[occupancy_machine]
        durations_machine = durations_machine[occupancy_machine]

        ending_times = schedule_machine + durations_machine
        previous_ending_times = ending_times.copy()
        previous_ending_times[1:] = ending_times[:-1]
        previous_ending_times[0] = 0

        starting_times_candidates = np.stack(
            (schedule_machine, previous_ending_times), axis=1
        )
        schedule_machine = np.max(starting_times_candidates, axis=1)

        schedule_machine = schedule_machine[np.argsort(occupancy_machine)]
        schedule[affectations == machine_id] = schedule_machine

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


def _occupancy(affectations: np.ndarray, schedule: np.ndarray) -> list:
    """Compute the occupancy of each machine of the given schedule.
    The occupancy of a machine is the order a machine treat each job.

    ---
    Returns:
        The occupancy of each machine. List of `n_machines` elements, where it is either
        an array containing the order of its scheduled jobs or None if no jobs are
        scheduled on this machine.
    """
    n_machines = affectations.shape[1]
    occupancy = []
    for machine_id in range(n_machines):
        if not np.any(affectations == machine_id):
            occupancy.append(None)
            continue

        machine_schedule = schedule[affectations == machine_id]
        occupancy.append(np.argsort(machine_schedule))

    return occupancy
