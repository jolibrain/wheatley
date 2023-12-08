import numpy as np
from einops import repeat


def validate_solution(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
):
    # Make sure the given args are valids.
    validate_instance(durations, affectations)
    assert schedule.shape == durations.shape, "Wrong starting_times shape"

    # Validate the solution.
    assert np.all(schedule != -1), "Not all tasks have been started!"
    validate_job_tasks(durations, affectations, schedule)
    validate_machine_tasks(durations, affectations, schedule)


def validate_instance(durations: np.ndarray, affectations: np.ndarray):
    n_jobs, n_machines = durations.shape
    assert durations.shape == affectations.shape, "Wrong number of jobs or machines"
    assert (
        affectations.min() == 0 and affectations.max() == n_machines - 1
    ), f"The indices must be in the range [0, n_machines - 1] (found [{affectations.min()}, {affectations.max()}])"
    ordered_index = np.arange(n_machines)
    assert np.all(
        np.sort(affectations, axis=1) == repeat(ordered_index, "m -> n m", n=n_jobs)
    ), "The machines are not all used once for each job"


def validate_job_tasks(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
):
    n_jobs, _ = durations.shape

    ending_times = schedule + durations
    previous_ending_times = np.concatenate(
        (np.zeros((n_jobs, 1), dtype=durations.dtype), ending_times[:, :-1]),
        axis=1,
    )
    assert np.all(
        schedule - previous_ending_times >= -1e-5
    ), "Some tasks starts before their precedence (job-wise) is finished"


def validate_machine_tasks(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
):
    _, n_machines = durations.shape

    for machine_id in range(n_machines):
        if not np.any(affectations == machine_id):
            continue

        schedule_machine = schedule[affectations == machine_id]
        durations_machine = durations[affectations == machine_id]

        sort_by_starting_times = np.argsort(schedule_machine)
        schedule_machine = schedule_machine[sort_by_starting_times]
        durations_machine = durations_machine[sort_by_starting_times]

        ending_times = schedule_machine + durations_machine
        previous_ending_times = ending_times.copy()
        previous_ending_times[1:] = ending_times[:-1]
        previous_ending_times[0] = 0

        assert np.all(
            schedule_machine - previous_ending_times >= -1e-5
        ), "Some tasks starts before their precedence (machine-wise) is finished"


def validate_solution_with_missing_tasks(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
):
    # Make sure the given args are valids.
    validate_instance_with_fictive_tasks(durations, affectations)
    assert schedule.shape == durations.shape, "Wrong starting_times shape"

    # Validate the solution.
    assert np.all(schedule != -1), "Not all tasks have been started!"
    validate_job_tasks(durations, affectations, schedule)
    validate_machine_tasks(durations, affectations, schedule)


def validate_instance_with_fictive_tasks(
    durations: np.ndarray,
    affectations: np.ndarray,
):
    n_machines = durations.shape[1]
    assert durations.shape == affectations.shape, "Wrong number of jobs or machines"
    assert (
        affectations.min() == 0 and affectations.max() == n_machines
    ), f"The indices must be in the range [0, n_machines] (found [{affectations.min()}, {affectations.max()}])"
