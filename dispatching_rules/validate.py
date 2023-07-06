import numpy as np
from einops import repeat


def validate_solution(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
):
    # Make sure the given args are valids.
    validate_instance(processing_times, machines)
    assert starting_times.shape == processing_times.shape, "Wrong starting_times shape"

    # Validate the solution.
    assert np.all(starting_times != -1), "Not all tasks have been started!"
    validate_job_tasks(processing_times, machines, starting_times)
    validate_machine_tasks(processing_times, machines, starting_times)


def validate_instance(processing_times: np.ndarray, machines: np.ndarray):
    n_jobs, n_machines = processing_times.shape
    assert processing_times.shape == machines.shape, "Wrong number of jobs or machines"
    assert (
        machines.min() == 0 and machines.max() == n_machines - 1
    ), "The indices must be in the range [0, n_machines - 1]"
    ordered_index = np.arange(n_machines)
    assert np.all(
        np.sort(machines, axis=1) == repeat(ordered_index, "m -> n m", n=n_jobs)
    ), "The machines are not all used once for each job"


def validate_job_tasks(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
):
    n_jobs, _ = processing_times.shape

    ending_times = starting_times + processing_times
    previous_ending_times = np.concatenate(
        (np.zeros((n_jobs, 1), dtype=np.int32), ending_times[:, :-1]),
        axis=1,
    )
    assert np.all(
        previous_ending_times <= starting_times
    ), "Some tasks starts before their precedence (job-wise) is finished"


def validate_machine_tasks(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
):
    _, n_machines = processing_times.shape

    sort_by_machines = np.argsort(machines, axis=1)
    processing_times = np.take_along_axis(processing_times, sort_by_machines, axis=1)
    starting_times = np.take_along_axis(starting_times, sort_by_machines, axis=1)

    processing_times = processing_times.transpose()
    starting_times = starting_times.transpose()

    sort_by_starting_times = np.argsort(starting_times, axis=1)
    starting_times = np.take_along_axis(starting_times, sort_by_starting_times, axis=1)
    processing_times = np.take_along_axis(
        processing_times, sort_by_starting_times, axis=1
    )

    ending_times = starting_times + processing_times
    previous_ending_times = np.concatenate(
        (np.zeros((n_machines, 1), dtype=np.int32), ending_times[:, :-1]),
        axis=1,
    )
    assert np.all(
        previous_ending_times <= starting_times
    ), "Some tasks starts before their precedence (machine-wise) is finished"
