from ortools.sat.python import cp_model
import tqdm
import time
from psp.solution import Solution


def interval_intersect(i1, i2):
    ret = []
    i = j = 0
    while i < len(i1) and j < len(i2):
        # check if i1 intersects i2
        low = max(i1[i][0], i2[j][0])
        high = min(i1[i][1], i2[j][1])
        if low <= high:
            ret.append((low, high))
        # remove interval with smallest endpoint
        if i1[i][1] < i2[j][1]:
            i += 1
        else:
            j += 1
    return ret


def solve_problem(problem, duration_param, max_time_ortools, optim_makespan=True):
    ntasks = len(problem.job_labels)

    if duration_param == "averagistic":
        task_to_dur = [problem.durations[0][t][0] for t in range(ntasks)]
    elif duration_param == "pessimistic":
        task_to_dur = [problem.durations[1][t][0] for t in range(ntasks)]
    else:  # duration_param == "optimistic":
        task_to_dur = [problem.durations[2][t][0] for t in range(ntasks)]

    nresources = problem.n_renewable_resources

    resource_to_capacity = problem.resource_availabilities

    task_and_resource_to_consumption = [
        [problem.resource_cons[t][0][r] for r in range(nresources)]
        for t in range(ntasks)
    ]

    task_successors = problem.successors_id
    task_release_date = [0] * ntasks

    # TODO faire la transformation ici pour indiquer pour chaque tache la liste de ses calendriers (intersection des calendriers des ressources utilisées par la tache)
    # task_to_cals = [[(0,horizon)] for t in range(ntasks)]
    task_to_cals = []
    horizon = 0
    for cals in problem.cals.values():
        end = cals[-1][1]  # end of last interval
        if end > horizon:
            horizon = end
    for t in range(ntasks):
        cals = [(0, horizon)]
        rc = problem.resource_cons[t][0]  # 0 for mode id, single mode ATM
        for i in range(problem.n_renewable_resources):
            if rc[i] != 0:
                cals = interval_intersect(cals, problem.cals[problem.res_cal[i]])
        task_to_cals.append(cals)

    status, starts = solve(
        ntasks,
        nresources,
        resource_to_capacity=resource_to_capacity,
        task_and_resource_to_consumption=task_and_resource_to_consumption,
        task_to_dur=task_to_dur,
        task_to_cals=task_to_cals,
        task_successors=task_successors,
        task_release_date=task_release_date,
        task_due_date=problem.due_dates,
        max_time_ortools=max_time_ortools,
        horizon=horizon,
        optim_makespan=optim_makespan,
    )

    return Solution(
        starts, [0] * len(starts), None, task_to_dur
    ), status == cp_model.OPTIMAL


def solve(
    ntasks,
    nresources,
    resource_to_capacity,
    task_and_resource_to_consumption,
    task_to_dur,
    task_to_cals,
    task_successors,
    task_release_date,
    task_due_date,
    max_time_ortools=0,
    horizon=10000,
    optim_makespan=True,
):
    # Create the model.
    model = cp_model.CpModel()

    # Variables containers.
    task_starts = {}
    task_ends = {}
    task_durations = {}
    task_intervals = {}
    task_cal_starts = {}
    task_cal_ends = {}
    task_cal_durations = {}
    task_cal_intervals = {}
    task_cal_presence = {}

    tmp_start_task_cal = {}
    tmp_end_task_cal = {}

    # Create makespan variable
    makespan = model.NewIntVar(0, horizon, "makespan")

    # tasks = range(len(problem.job_labels))
    tasks = range(ntasks)

    for t in tqdm.tqdm(tasks, desc="adding var and constraint for tasks", leave=False):
        # Create variables
        start_var = model.NewIntVar(task_release_date[t], horizon, f"start_of_task_{t}")
        end_var = model.NewIntVar(0, horizon, f"end_of_task_{t}")
        dur_var = model.NewIntVar(0, horizon, f"duration_of_task{t}")
        interval_var = model.NewIntervalVar(
            start_var, dur_var, end_var, f"task_interval_{t}"
        )

        # Store task variables.
        task_starts[t] = start_var
        task_ends[t] = end_var
        task_durations[t] = dur_var
        task_intervals[t] = interval_var

        # Add constraint makespan
        model.Add(end_var <= makespan)

        model.Add(dur_var >= task_to_dur[t])

        for cal in task_to_cals[t]:
            (start, end) = cal
            start_cal_var = model.NewIntVar(
                start, end, f"start_of_task_{t}_in_cal_{cal}"
            )
            end_cal_var = model.NewIntVar(start, end, f"end_of_task_{t}_in_cal_{cal}")
            dur_cal_var = model.NewIntVar(
                0, end - start, f"duration_of_task{t}_in_cal_{cal}"
            )
            task_cal_pres_var = model.NewBoolVar(
                f"presence_task_interval_{t}_in_cal_{cal}"
            )
            interval_cal_var = model.NewOptionalIntervalVar(
                start_cal_var,
                dur_cal_var,
                end_cal_var,
                task_cal_pres_var,
                f"task_interval_{t}_in_cal_{cal}",
            )

            task_cal_starts[(t, cal)] = start_cal_var
            task_cal_ends[(t, cal)] = end_cal_var
            task_cal_durations[(t, cal)] = dur_cal_var
            task_cal_intervals[(t, cal)] = interval_cal_var
            task_cal_presence[(t, cal)] = task_cal_pres_var

            tmp_start_task_cal_var = model.NewIntVar(
                start, horizon, f"tmp_start_of_task_{t}_in_cal_{cal}"
            )
            tmp_end_task_cal_var = model.NewIntVar(
                0, end, f"tmp_end_of_task_{t}_in_cal_{cal}"
            )

            tmp_start_task_cal[(t, cal)] = tmp_start_task_cal_var
            tmp_end_task_cal[(t, cal)] = tmp_end_task_cal_var

            # Make duration =0 if not present
            model.Add(dur_cal_var == 0).OnlyEnforceIf(~task_cal_pres_var)

            if task_to_dur[t] > 0:
                model.Add(dur_cal_var > 0).OnlyEnforceIf(task_cal_pres_var)

            # tmp_start is equal to start var only if present, equals horizon otherwise
            model.Add(tmp_start_task_cal_var == start_cal_var).OnlyEnforceIf(
                task_cal_pres_var
            )

            model.Add(tmp_start_task_cal_var == horizon).OnlyEnforceIf(
                ~task_cal_pres_var
            )

            # tmp_end is equal to end var only if present, equals 0 otherwise
            model.Add(tmp_end_task_cal_var == end_cal_var).OnlyEnforceIf(
                task_cal_pres_var
            )

            model.Add(tmp_end_task_cal_var == 0).OnlyEnforceIf(~task_cal_pres_var)

        # Ensures that startOf(itv_t) == min[cal of t] startOf(itv_{t,cal})
        model.AddMinEquality(
            start_var, [tmp_start_task_cal[(t, cal)] for cal in task_to_cals[t]]
        )

        # Ensures that endOf(itv_t) == max[cal of t] endOf(itv_{t,cal})
        model.AddMaxEquality(
            end_var, [tmp_end_task_cal[(t, cal)] for cal in task_to_cals[t]]
        )

        # Sum(duration_cal) = desired_duration
        model.Add(
            sum([task_cal_durations[(t, cal)] for cal in task_to_cals[t]])
            == task_to_dur[t]
        )

        for cal_index in range(len(task_to_cals[t]) - 1):
            cal = task_to_cals[t][cal_index]
            next_cal = task_to_cals[t][cal_index + 1]
            (start, end) = cal
            model.Add(task_cal_ends[(t, cal)] >= end * task_cal_presence[(t, next_cal)])

        for cal_index in range(1, len(task_to_cals[t])):
            cal = task_to_cals[t][cal_index]
            prev_cal = task_to_cals[t][cal_index - 1]
            (start, end) = cal
            model.Add(
                task_cal_starts[(t, cal)]
                <= start * task_cal_presence[(t, prev_cal)]
                + horizon * (1 - task_cal_presence[(t, prev_cal)])
            )

    # Precedence constraints
    for t in tasks:
        for u in task_successors[t]:
            model.Add(task_starts[u] >= task_ends[t])

    # Cumulative constraints
    for r in range(nresources):
        tasks_r = [t for t in tasks if task_and_resource_to_consumption[t][r] > 0]
        # print(f"tasks_{r} = {tasks_r}")
        model.AddCumulative(
            [task_intervals[t] for t in tasks_r],
            [task_and_resource_to_consumption[t][r] for t in tasks_r],
            resource_to_capacity[r],
        )

    # Objective.
    if optim_makespan:
        objective = makespan
    else:
        tardiness_tasks = [
            max([0, task_ends[t] - task_due_date[t]]) for t in range(tasks)
        ]
        objective = sum(tardiness_tasks)

    model.Minimize(objective)

    # Solve model.
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = max_time_ortools
    # if in_main_solve:
    solver.parameters.log_search_progress = True
    solver.parameters.symmetry_level = 4
    solver.parameters.num_workers = 3

    start_time = time.time()
    status = solver.Solve(model)
    print("ortools solve time: ", time.time() - start_time)
    # if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    #     print("optimal")
    #     print(f"makespan={solver.value(makespan)}")
    #     for t in tasks:
    #         print(f"task {t}: ({solver.value(task_starts[t])},{solver.value(task_ends[t])})")
    #         for cal in task_to_cals[t]:
    #             print(f"\t cal {cal} : {solver.value(task_cal_presence[(t,cal)])} - ({solver.value(task_cal_starts[(t,cal)])},{solver.value(task_cal_ends[(t,cal)])})")

    # else:
    #     print("UNSAT")
    if status == cp_model.OPTIMAL:
        print("ortools found optimal solution")
    elif status == cp_model.FEASIBLE:
        print("ortools found some solution")
        print("Solver bound: ", solver.best_objective_bound)
    else:
        print("Ortools could not find any solution")

    # for t in tasks:
    #     print("t", t)
    #     print("start", solver.value(task_starts[t]))
    return status, [solver.value(task_starts[t]) for t in tasks]


if __name__ == "__main__":
    ntasks = 3

    # Inputs
    nresources = 2
    resource_to_capacity = [1, 1]
    task_and_resource_to_consumption = [[1, 0], [1, 1], [0, 1]]
    task_to_dur = [2, 2, 3]
    task_to_cals = [
        [(0, 1), (2, 3), (4, 5), (6, 20)],
        [(1, 2), (3, 4), (6, 30)],
        [(0, 1), (2, 3), (4, 5), (6, 30)],
    ]
    task_successors = [[], [], []]
    task_release_date = [0, 0, 0]
    task_due_date = [30, 30, 30]

    solve(
        ntasks,
        nresources,
        resource_to_capacity=resource_to_capacity,
        task_and_resource_to_consumption=task_and_resource_to_consumption,
        task_to_dur=task_to_dur,
        task_to_cals=task_to_cals,
        task_successors=task_successors,
        task_release_date=task_release_date,
        task_due_date=task_due_date,
        max_time_ortools=30,
        horizon=30,
    )
