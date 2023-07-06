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
    ), "The indices must be in the range [0, n_machines - 1]"
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
        (np.zeros((n_jobs, 1), dtype=np.int32), ending_times[:, :-1]),
        axis=1,
    )
    assert np.all(
        previous_ending_times <= schedule
    ), "Some tasks starts before their precedence (job-wise) is finished"


def validate_machine_tasks(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
):
    _, n_machines = durations.shape

    sort_by_machines = np.argsort(affectations, axis=1)
    durations = np.take_along_axis(durations, sort_by_machines, axis=1)
    schedule = np.take_along_axis(schedule, sort_by_machines, axis=1)

    durations = durations.transpose()
    schedule = schedule.transpose()

    sort_by_starting_times = np.argsort(schedule, axis=1)
    schedule = np.take_along_axis(schedule, sort_by_starting_times, axis=1)
    durations = np.take_along_axis(durations, sort_by_starting_times, axis=1)

    ending_times = schedule + durations
    previous_ending_times = np.concatenate(
        (np.zeros((n_machines, 1), dtype=np.int32), ending_times[:, :-1]),
        axis=1,
    )
    assert np.all(
        previous_ending_times <= schedule
    ), "Some tasks starts before their precedence (machine-wise) is finished"
