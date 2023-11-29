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
        if np.any(durations == -1):
            _, n_machines = affectations.shape
            # There are fictive tasks.

            # Set the fictive tasks as a task on a fictive machine
            # with a duration of 0.
            durations = durations.copy()
            durations[durations == -1] = 0

            affectations = affectations.copy()
            affectations[affectations == -1] = n_machines
            # TODO: Add validation.
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
    # TODO: missing tasks
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


def missing_tasks_to_fictive(durations: np.ndarray, affectations: np.ndarray):
    """Replace missing tasks with fictive ones, so that each job has a task on all machines.
    A fictive task has a duration of 0, so it can be placed as soon as possible
    and does not impact the final scheduling value.

    A missing task is expected to have its machine id and duration equal to -1.
    """
    n_jobs, n_machines = affectations.shape
    job_range = np.arange(n_jobs)
    sort_by_machines = np.argsort(affectations, axis=1)

    missing_tasks_ids = np.zeros(n_jobs, dtype=int)
    existing_tasks_ids = np.ones(n_jobs, dtype=int) * (n_machines - 1)

    for rank in range(n_machines - 1, -1, -1):
        affectation_ids = sort_by_machines[job_range, existing_tasks_ids]
        machine_ids = affectations[job_range, affectation_ids]
        valid_tasks = machine_ids == rank

        missing_machine_ids = sort_by_machines[job_range, missing_tasks_ids]
        machine_ids = affectations[job_range, missing_machine_ids]
        affectations[job_range, missing_machine_ids] = (machine_ids * valid_tasks) + (
            rank * ~valid_tasks
        )

        missing_tasks_ids[~valid_tasks] += 1
        existing_tasks_ids[valid_tasks] -= 1

    durations[durations == -1] = 0

    assert np.all(missing_tasks_ids - existing_tasks_ids == np.ones(n_jobs, dtype=int))
    assert np.all(durations != -1) and np.all(affectations != -1)
    machine_ids = repeat(np.arange(n_machines), "m -> j m", j=n_jobs)
    assert np.all(np.sort(affectations, axis=1) == machine_ids)
