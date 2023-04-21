from utils.loaders import PSPLoader
from utils.ortools import solve_psp, compute_ortools_makespan_on_real_duration
import numpy as np


def test_ortools_small():
    psp = PSPLoader().load_single("instances/psp/small/small.sm")
    durations = [item for sublist in psp["durations"] for item in sublist]
    sol, optimal = solve_psp(psp, durations, 3, 1)
    assert sol.job_schedule == [0, 0, 0, 4, 4, 10, 11, 15]
    assert sol.modes == [0] * 8
    assert sol.get_makespan() == 15
    assert optimal == True


def test_ortools_uncertainty(state_small):
    psp = PSPLoader().load_single("instances/psp/small/small.sm")
    durations = [item for sublist in psp["durations"] for item in sublist]
    sol, optimal = solve_psp(psp, durations, 3, 1)
    assert sol.job_schedule == [0, 0, 0, 4, 4, 10, 11, 15]

    state = state_small
    state.problem["durations"] = [[0], [5], [4], [6], [2], [1], [4], [0]]
    state.reset_durations(redraw_real=True)

    assert np.all(
        np.equal(
            state.real_durations, np.array([0.0, 5.0, 4.0, 6.0, 2.0, 1.0, 4.0, 0.0])
        )
    )
    real_makespan, starts = compute_ortools_makespan_on_real_duration(sol, state_small)
    assert real_makespan == 16.0
