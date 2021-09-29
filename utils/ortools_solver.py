# Copy of https://developers.google.com/optimization/scheduling/job_shop#entire-program

import collections

import numpy as np
from ortools.sat.python import cp_model

from problem.solution import Solution

from config import MAX_TIME_ORTOOLS


def solve_jssp(affectations, durations):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = []
    for i in range(affectations.shape[0]):
        jobs_data.append([])
        for j in range(affectations.shape[1]):
            jobs_data[-1].append((int(affectations[i, j]), int(durations[i, j])))

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = "_%i_%i" % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, "interval" + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_TIME_ORTOOLS
    # status = solver.Solve(model)

    schedule = np.zeros_like(affectations)
    # if status == cp_model.OPTIMAL:
    # Create one list of assigned tasks per machine.
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            schedule[job_id, task_id] = solver.Value(all_tasks[job_id, task_id].start)
    return Solution(schedule)

    # else:
    #     print("No Optimal solution found")
    #     return


if __name__ == "__main__":
    print(
        solve_jssp(
            np.array([[0, 1, 2], [2, 0, 1], [1, 0, 2]]),
            np.array([[2, 2, 4], [2, 4, 5], [2, 2, 6]]),
        )
    )
