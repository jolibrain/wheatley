from itertools import product

import numpy as np
import pytest

from .solver import Solver
from .validate import validate_job_tasks, validate_machine_tasks


@pytest.mark.parametrize(
    "durations, schedule, heuristic, ignore_unfinished_precedences, expected_schedule",
    [
        (
            np.array([[12, 20], [5, 13]]),
            np.array([[1, 0], [0, 1]]),
            "SFT",
            True,
            np.array([[0, 12], [0, 12]]),
        ),
        (
            np.array([[12, 20], [5, 11]]),
            np.array([[1, 0], [0, 1]]),
            "SFT",
            False,
            np.array([[16, 28], [0, 5]]),
        ),
        (
            np.array([[12, 20], [5, 13]]),
            np.array([[0, 1], [0, 1]]),
            "SFT",
            True,
            np.array([[5, 18], [0, 5]]),
        ),
        (
            np.array([[12, 20], [5, 13]]),
            np.array([[0, 1], [0, 1]]),
            "SFT",
            False,
            np.array([[5, 18], [0, 5]]),
        ),
        (
            np.array([[12, 20, 6], [5, 13, 20]]),
            np.array([[0, 2, 1], [0, 1, 2]]),
            "SFT",
            True,
            np.array([[5, 17, 37], [0, 5, 37]]),
        ),
        (
            np.array([[12, 15, 8], [3, 11, 17], [8, 9, 10]]),
            np.array([[1, 0, 2], [1, 2, 0], [2, 0, 1]]),
            "SFT",
            True,
            np.array([[3, 17, 32], [0, 8, 32], [0, 8, 17]]),
        ),
    ],
)
def test_solver(
    durations: np.ndarray,
    schedule: np.ndarray,
    heuristic: str,
    ignore_unfinished_precedences: bool,
    expected_schedule: np.ndarray,
):
    solver = Solver(durations, schedule, heuristic, ignore_unfinished_precedences)
    schedule = solver.solve()
    assert np.all(schedule == expected_schedule)


@pytest.mark.parametrize(
    "durations, schedule",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 1, 2], [0, 1, 2]]),
        ),
        (
            np.array([[1, 1, 1], [1, 1, 1]]),
            np.array([[0, 1, 2], [0, 1, 2]]),
        ),
        (
            np.array([[3, 1, 1], [1, 1, 1]]),
            np.array([[0, 3, 4], [0, 1, 2]]),
        ),
        (
            np.array([[3, 1, 1], [3, 1, 1], [1, 1, 1]]),
            np.array([[0, 4, 5], [0, 4, 5], [0, 1, 2]]),
        ),
        (
            np.array([[3, 1, 1], [3, 1, 1], [1, 1, 1]]),
            np.array([[0, 2, 5], [0, 2, 5], [0, 1, 2]]),
        ),
    ],
)
def test_validate_job_tasks(durations: np.ndarray, schedule: np.ndarray):
    def simple_implementation(durations: np.ndarray, schedule: np.ndarray) -> bool:
        n_jobs, n_machines = durations.shape
        is_valid = True
        for job_id, machine_id in product(range(n_jobs), range(n_machines)):
            if machine_id == 0:
                continue

            previous_ending_time = (
                durations[job_id, machine_id - 1] + schedule[job_id, machine_id - 1]
            )
            starting_time = schedule[job_id, machine_id]
            if previous_ending_time > starting_time:
                is_valid = False

        return is_valid

    affectations = np.zeros_like(durations)
    is_valid = simple_implementation(durations, schedule)
    try:
        validate_job_tasks(durations, affectations, schedule)
        assert is_valid
    except AssertionError:
        assert not is_valid


@pytest.mark.parametrize(
    "durations, affectations, schedule",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 1, 2], [0, 1, 2]]),
            np.array([[0, 1, 2], [0, 1, 2]]),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 1, 2], [0, 1, 2]]),
            np.array([[0, 1, 2], [0, 1, 2]]),
        ),
    ],
)
def test_validate_machine_tasks(
    durations: np.ndarray, affectations: np.ndarray, schedule: np.ndarray
):
    def simple_implementation(
        durations: np.ndarray, affectations: np.ndarray, schedule: np.ndarray
    ) -> bool:
        n_jobs, n_machines = durations.shape
        machine_schedule = [[] for _ in range(n_machines)]
        is_valid = True

        for job_id, machine_id in product(range(n_jobs), range(n_machines)):
            starting_time = schedule[job_id, machine_id]
            ending_time = starting_time + durations[job_id, machine_id]
            machine_schedule[affectations[job_id, machine_id]].append(
                (starting_time, ending_time)
            )

        for schedule in machine_schedule:
            schedule = sorted(schedule, key=lambda x: x[0])
            for i in range(len(schedule) - 1):
                if schedule[i][1] > schedule[i + 1][0]:
                    is_valid = False

        return is_valid

    is_valid = simple_implementation(durations, affectations, schedule)
    try:
        validate_machine_tasks(durations, affectations, schedule)
        assert is_valid
    except AssertionError:
        assert not is_valid
