#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#
# This file is largely inspired from
# https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
# which comes with Copyright 2010-2022 Google LLC under Apache License, Version 2.0

import collections
from copy import deepcopy

import numpy as np
from ortools.sat.python import cp_model
from psp.utils.ortools_rcpsp_calendars import solve_problem as solve_problem_cal

from psp.solution import Solution

from psp.env.genv import GEnv

import torch


def node_from_job_mode(problem, jobid, modeid):
    nid = 0
    if isinstance(problem, dict):
        for i in range(jobid):
            nid += problem["job_info"][i][0]
    else:
        for i in range(jobid):
            nid += problem.n_modes_per_job[i]
    return nid + modeid


def compute_ortools_criterion_on_real_duration(
    solution, state, problem, criterion="makespan"
):
    state.ff = False  # disable source fast forward
    state.reset()  # reset do not redraw real durations

    aff = solution.job_schedule
    computed_start_dates = np.full_like(aff, -1.0)
    while True:
        if aff.min() == float("inf"):
            break

        datemin = np.where(aff == aff.min(), True, False)
        selectables = np.where(state.selectables() == 1, True, False)
        dateminAndSelectable = np.logical_and(datemin, selectables)
        index = np.argmax(dateminAndSelectable)  # get first
        modeid = solution.modes[index]
        aff[index] = float("inf")
        computed_start_dates[index] = state.affect_job(
            node_from_job_mode(state.problem, index, modeid)
        )[3]

    # with ortools, sink as last node is mandatory
    if criterion == "makespan":
        return state.tct_real(-1), computed_start_dates
    elif criterion == "tardiness":
        if problem.due_dates is None:
            print("asked for tardiness but no die dates given")
            return state.tct_real(-1), computed_start_dates
        tcts = state.all_tct_real()
        tardiness = 0.0
        for t, d in enumerate(problem.due_dates):
            if d is not None:
                tardiness += tcts[t] - d
        return tardiness, computed_start_dates


def get_ortools_criterion_psp(
    env,
    max_time_ortools,
    scaling_constant_ortools,
    ortools_strategy="pessimistic",
    criterion="makespan",
):
    if ortools_strategy == "realistic":
        durations = env.state.all_duration_real()
    elif ortools_strategy == "pessimistic":
        durations = env.state.all_durations()[:, 1]
    elif ortools_strategy == "optimistic":
        durations = env.state.all_durations()[:, 2]
    elif ortools_strategy == "averagistic":
        durations = env.state.all_durations()[:, 0]
    else:
        print("unknow ortools strategy ", ortools_strategy)
        exit()

    if isinstance(env, GEnv):
        durations = durations.numpy()
    if env.problem.res_cal is not None or criterion == "tardiness":
        solution, optimal = solve_problem_cal(
            env.problem,
            ortools_strategy,
            max_time_ortools,
            optim_makespan=(criterion == "makespan"),
        )
    else:
        solution, optimal = solve_psp(
            env.problem, durations, max_time_ortools, scaling_constant_ortools
        )
    if (
        env.state.deterministic
        and env.problem.res_cal is None
        and criterion == "makespan"
    ):
        return solution.get_criterion(), solution.schedule, optimal

    print(f"recomputing {criterion} on real env ", end="")
    real_criterion, starts = compute_ortools_criterion_on_real_duration(
        solution, env.state, env.problem, criterion=criterion
    )
    print("    ", real_criterion.item())
    torch.set_printoptions(threshold=10000)
    solution2 = Solution.from_mode_schedule(
        starts,
        env.state.problem,
        env.state.all_affected(),
        env.state.all_jobid(),
        real_durations=env.state.real_durations,
        criterion=real_criterion,
    )
    return real_criterion, solution2.schedule, optimal


def solve_psp(problem, durations, max_time_ortools, scaling_constant_ortools):
    durations = (durations * scaling_constant_ortools).astype(int)

    # update problem durations with real_durations

    problem = deepcopy(problem)
    i = 0
    if isinstance(problem, dict):
        for j in range(len(problem["durations"][0])):
            for m in range(len(problem["durations"][0][j])):
                problem["durations"][0][j][m] = durations[i]
                i += 1
    else:
        for j in range(len(problem.durations[0])):
            for m in range(len(problem.durations[0][j])):
                problem.durations[0][j][m] = durations[i]
                i += 1

    # return solution, is_Optimal
    intervals_of_tasks, after = AnalyseDependencyGraph(problem)
    delays, initial_solution, optimal_found = ComputeDelaysBetweenNodes(
        problem, intervals_of_tasks
    )
    last_task = problem["n_jobs"] if isinstance(problem, dict) else problem.n_jobs
    key = (0, last_task)
    lower_bound = delays[key][0] if key in delays else 0

    bound, value, assignment, optimal = SolveRcpsp(
        problem=problem,
        proto_file="",
        params="",
        active_tasks=set(range(1, last_task + 1)),
        source=0,
        sink=last_task,
        intervals_of_tasks=intervals_of_tasks,
        delays=delays,
        in_main_solve=True,
        initial_solution=initial_solution,
        lower_bound=lower_bound,
        max_time_ortools=max_time_ortools,
    )

    return (
        Solution(
            problem,
            np.array([e / scaling_constant_ortools for e in assignment[0]]),
            np.array(assignment[1]),
            None,
            np.array([d / scaling_constant_ortools for d in durations]),
            criterion=value / scaling_constant_ortools,
        ),
        optimal,
    )


def AnalyseDependencyGraph(problem):
    """Analyses the dependency graph to improve the model.
    Args:
      problem: the protobuf of the problem to solve.
    Returns:
      a list of (task1, task2, in_between_tasks) with task2 and indirect successor
      of task1, and in_between_tasks being the list of all tasks after task1 and
      before task2.
    """

    if isinstance(problem, dict):
        num_nodes = len(problem["job_info"])
    else:
        num_nodes = problem.n_jobs

    ins = collections.defaultdict(list)
    outs = collections.defaultdict(list)
    after = collections.defaultdict(set)
    before = collections.defaultdict(set)

    # Build the transitive closure of the precedences.
    # This algorithm has the wrong complexity (n^4), but is OK for the psplib
    # as the biggest example has 120 nodes.
    if isinstance(problem, dict):
        for n in range(num_nodes):
            for s in problem["job_info"][n][1]:
                ins[s].append(n + 1)
                outs[n + 1].append(s)

                for a in list(after[s]) + [s]:
                    for b in list(before[n + 1]) + [n + 1]:
                        after[b].add(a)
                        before[a].add(b)
    else:
        for n in range(num_nodes):
            for s in problem.successors_id[n]:
                # ortools uses legacy job ids that start at 1
                s += 1
                ins[s].append(n + 1)
                outs[n + 1].append(s)

                for a in list(after[s]) + [s]:
                    for b in list(before[n + 1]) + [n + 1]:
                        after[b].add(a)
                        before[a].add(b)

    # Search for pair of tasks, containing at least two parallel branch between
    # them in the precedence graph.
    num_candidates = 0
    result = []
    for source, start_outs in outs.items():
        if len(start_outs) <= 1:
            # Starting with the unique successor of source will be as good.
            continue
        for sink, end_ins in ins.items():
            if len(end_ins) <= 1:
                # Ending with the unique predecessor of sink will be as good.
                continue
            if sink == source:
                continue
            if sink not in after[source]:
                continue

            num_active_outgoing_branches = 0
            num_active_incoming_branches = 0
            for succ in outs[source]:
                if sink in after[succ]:
                    num_active_outgoing_branches += 1
            for pred in ins[sink]:
                if source in before[pred]:
                    num_active_incoming_branches += 1

            if num_active_outgoing_branches <= 1 or num_active_incoming_branches <= 1:
                continue

            common = after[source].intersection(before[sink])
            if len(common) <= 1:
                continue
            num_candidates += 1
            result.append((source, sink, common))

    # Sort entries lexicographically by (len(common), source, sink)
    def Price(entry):
        return num_nodes * num_nodes * len(entry[2]) + num_nodes * entry[0] + entry[1]

    result.sort(key=Price)
    # print(f"  - created {len(result)} pairs of nodes to examine", flush=True)
    return result, after


def SolveRcpsp(
    problem,
    proto_file,
    params,
    active_tasks,
    source,
    sink,
    intervals_of_tasks,
    delays,
    in_main_solve=False,
    initial_solution=None,
    lower_bound=0,
    max_time_ortools=0,
):
    """Parse and solve a given RCPSP problem in proto format.
    The model will only look at the tasks {source} + {sink} + active_tasks, and
    ignore all others.
    Args:
      problem: the description of the model to solve in protobuf format
      proto_file: the name of the file to export the CpModel proto to.
      params: the string representation of the parameters to pass to the sat
        solver.
      active_tasks: the set of active tasks to consider.
      source: the source task in the graph. Its end will be forced to 0.
      sink: the sink task of the graph. Its start is the makespan of the problem.
      intervals_of_tasks: a heuristic lists of (task1, task2, tasks) used to add
        redundant energetic equations to the model.
      delays: a list of (task1, task2, min_delays) used to add extended precedence
        constraints (start(task2) >= end(task1) + min_delay).
      in_main_solve: indicates if this is the main solve procedure.
      initial_solution: A valid assignment used to hint the search.
      lower_bound: A valid lower bound of the makespan objective.
    Returns:
      (lower_bound of the objective, best solution found, asssignment)
    """
    # Create the model.

    model = cp_model.CpModel()

    if isinstance(problem, dict):
        num_resources = problem["n_resources"]
    else:
        num_resources = problem.n_resources

    all_active_tasks = list(active_tasks)
    all_active_tasks.sort()
    all_resources = range(num_resources)

    horizon = -1
    if delays and in_main_solve and (source, sink) in delays:
        horizon = delays[(source, sink)][1]
    elif horizon == -1:  # Naive computation.
        # horizon = sum(max(r.duration for r in t.recipes) for t in problem.tasks)
        if isinstance(problem, dict):
            horizon = sum([max(t) for t in problem["durations"][0]])
        else:
            horizon = sum([max(t) for t in problem.durations[0]])
    # if in_main_solve:
    # print(f"Horizon = {horizon}", flush=True)

    # Containers.
    task_starts = {}
    task_ends = {}
    task_durations = {}
    task_intervals = {}
    task_resource_to_energy = {}
    task_to_resource_demands = collections.defaultdict(list)

    task_to_presence_literals = collections.defaultdict(list)
    task_to_recipe_durations = collections.defaultdict(list)
    task_resource_to_fixed_demands = collections.defaultdict(dict)
    task_resource_to_max_energy = collections.defaultdict(int)

    resource_to_sum_of_demand_max = collections.defaultdict(int)

    # Create task variables.
    for t in all_active_tasks:
        # task = problem.tasks[t]
        # num_recipes = len(task.recipes)
        if isinstance(problem, dict):
            num_recipes = len(problem["durations"][0][t - 1])
        else:
            num_recipes = len(problem.durations[0][t - 1])
        all_recipes = range(num_recipes)

        start_var = model.NewIntVar(0, horizon, f"start_of_task_{t}")
        end_var = model.NewIntVar(0, horizon, f"end_of_task_{t}")

        literals = []
        if num_recipes > 1:
            # Create one literal per recipe.
            literals = [model.NewBoolVar(f"is_present_{t}_{r}") for r in all_recipes]

            # Exactly one recipe must be performed.
            model.AddExactlyOne(literals)

        else:
            literals = [1]

        # Temporary data structure to fill in 0 demands.
        demand_matrix = collections.defaultdict(int)

        # Scan recipes and build the demand matrix and the vector of durations.
        # for recipe_index, recipe in enumerate(task.recipes):
        #     task_to_recipe_durations[t].append(recipe.duration)
        #     for demand, resource in zip(recipe.demands, recipe.resources):
        #         demand_matrix[(resource, recipe_index)] = demand
        if isinstance(problem, dict):
            for m in range(len(problem["durations"][0][t - 1])):
                task_to_recipe_durations[t].append(problem["durations"][0][t - 1][m])
                for r in range(problem["n_resources"]):
                    if problem["resources"][t - 1][m][r] != 0:
                        demand_matrix[(r, m)] = problem["resources"][t - 1][m][r]
        else:
            for m in range(len(problem.durations[0][t - 1])):
                task_to_recipe_durations[t].append(problem.durations[0][t - 1][m])
                for r in range(problem.n_resources):
                    if problem.resource_cons[t - 1][m][r] != 0:
                        demand_matrix[(r, m)] = problem.resource_cons[t - 1][m][r]

        # Create the duration variable from the accumulated durations.
        duration_var = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(task_to_recipe_durations[t]),
            f"duration_of_task_{t}",
        )

        # Link the recipe literals and the duration_var.
        for r in range(num_recipes):
            model.Add(duration_var == task_to_recipe_durations[t][r]).OnlyEnforceIf(
                literals[r]
            )

        # Create the interval of the task.
        task_interval = model.NewIntervalVar(
            start_var, duration_var, end_var, f"task_interval_{t}"
        )

        # Store task variables.
        task_starts[t] = start_var
        task_ends[t] = end_var
        task_durations[t] = duration_var
        task_intervals[t] = task_interval
        task_to_presence_literals[t] = literals

        # Create the demand variable of the task for each resource.
        for res in all_resources:
            demands = [demand_matrix[(res, recipe)] for recipe in all_recipes]
            task_resource_to_fixed_demands[(t, res)] = demands
            demand_var = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(demands), f"demand_{t}_{res}"
            )
            task_to_resource_demands[t].append(demand_var)

            # Link the recipe literals and the demand_var.
            for r in all_recipes:
                model.Add(demand_var == demand_matrix[(res, r)]).OnlyEnforceIf(
                    literals[r]
                )

            resource_to_sum_of_demand_max[res] += max(demands)

        # Create the energy expression for (task, resource):
        for res in all_resources:
            task_resource_to_energy[(t, res)] = sum(
                literals[r]
                * task_to_recipe_durations[t][r]
                * task_resource_to_fixed_demands[(t, res)][r]
                for r in all_recipes
            )
            task_resource_to_max_energy[(t, res)] = max(
                task_to_recipe_durations[t][r]
                * task_resource_to_fixed_demands[(t, res)][r]
                for r in all_recipes
            )

    # Create makespan variable
    makespan = model.NewIntVar(lower_bound, horizon, "makespan")
    makespan_size = model.NewIntVar(1, horizon, "interval_makespan_size")
    interval_makespan = model.NewIntervalVar(
        makespan, makespan_size, model.NewConstant(horizon + 1), "interval_makespan"
    )

    # Normal dependencies (task ends before the start of successors).
    if isinstance(problem, dict):
        for t in all_active_tasks:
            # for n in problem.tasks[t].successors:
            for n in problem["job_info"][t - 1][1]:
                if n == sink:
                    model.Add(task_ends[t] <= makespan)
                elif n in active_tasks:
                    model.Add(task_ends[t] <= task_starts[n])
    else:
        for t in all_active_tasks:
            # for n in problem.tasks[t].successors:
            for n in problem.successors_id[t - 1]:
                # ortools uses legacy job ids that start at 1
                n += 1
                if n == sink:
                    model.Add(task_ends[t] <= makespan)
                elif n in active_tasks:
                    model.Add(task_ends[t] <= task_starts[n])

    # Containers for resource investment problems.
    capacities = []  # Capacity variables for all resources.
    max_cost = 0  # Upper bound on the investment cost.

    # Create resources.
    for res in all_resources:
        # resource = problem.resources[res]
        # c = resource.max_capacity
        if isinstance(problem, dict):
            c = problem["resource_availability"][res]
        else:
            c = problem.resource_availabilities[res]
        if c == -1:
            print(f"No capacity: {resource}")
            c = resource_to_sum_of_demand_max[res]

        # RIP problems have only renewable resources, and no makespan.
        # if problem.is_resource_investment or resource.renewable:
        if isinstance(problem, dict):
            nren = problem["n_renewable_resources"]
        else:
            nren = problem.n_renewable_resources
        if res < nren:
            intervals = [task_intervals[t] for t in all_active_tasks]
            demands = [task_to_resource_demands[t][res] for t in all_active_tasks]

            # if _USE_INTERVAL_MAKESPAN.value:
            if True:
                intervals.append(interval_makespan)
                demands.append(c)

                model.AddCumulative(intervals, demands, c)
        else:  # Non empty non renewable resource. (single mode only)
            model.Add(
                cp_model.LinearExpr.Sum(
                    [task_to_resource_demands[t][res] for t in all_active_tasks]
                )
                <= c
            )

    # Objective.
    objective = makespan

    model.Minimize(objective)

    # Add min delay constraints.
    if delays is not None:
        for (local_start, local_end), (min_delay, _) in delays.items():
            if local_start == source and local_end in active_tasks:
                model.Add(task_starts[local_end] >= min_delay)
            elif local_start in active_tasks and local_end == sink:
                model.Add(makespan >= task_ends[local_start] + min_delay)
            elif local_start in active_tasks and local_end in active_tasks:
                model.Add(task_starts[local_end] >= task_ends[local_start] + min_delay)

    problem_is_single_mode = True
    for t in all_active_tasks:
        if len(task_to_presence_literals[t]) > 1:
            problem_is_single_mode = False
            break

    # Add sentinels.
    task_starts[source] = 0
    task_ends[source] = 0
    task_to_presence_literals[0].append(True)
    task_starts[sink] = makespan
    task_to_presence_literals[sink].append(True)

    # Add solution hint.
    if initial_solution:
        for t in all_active_tasks:
            model.AddHint(task_starts[t], initial_solution.start_of_task[t])
            if len(task_to_presence_literals[t]) > 1:
                selected = initial_solution.selected_recipe_of_task[t]
                model.AddHint(task_to_presence_literals[t][selected], 1)

    # Write model to file.
    if proto_file:
        print(f"Writing proto to{proto_file}")
        model.ExportToFile(proto_file)

    # Solve model.
    solver = cp_model.CpSolver()
    # if params:
    #     text_format.Parse(params, solver.parameters)
    if not in_main_solve:
        solver.parameters.num_search_workers = 16
        solver.parameters.max_time_in_seconds = 0.0
    else:
        solver.parameters.max_time_in_seconds = max_time_ortools
    # if in_main_solve:
    #     solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # assignment = rcpsp_pb2.RcpspAssignment()
        assignment_start = []
        assignment_mode = []
        # for t in range(len(problem.tasks)):
        if isinstance(problem, dict):
            ntasks = len(problem["job_info"])
        else:
            ntasks = problem.n_modes
        for t in range(1, ntasks + 1):
            if t in task_starts:
                # assignment.start_of_task.append(solver.Value(task_starts[t]))
                assignment_start.append(solver.Value(task_starts[t]))
                for r in range(len(task_to_presence_literals[t])):
                    if solver.BooleanValue(task_to_presence_literals[t][r]):
                        # assignment.selected_recipe_of_task.append(r)
                        assignment_mode.append(r)
                        break
            else:  # t is not an active task.
                # assignment.start_of_task.append(0)
                assignment_start.append(0)
                # assignment.selected_recipe_of_task.append(0)
                assignment_mode.append(0)
        return (
            int(solver.BestObjectiveBound()),
            int(solver.ObjectiveValue()),
            (assignment_start, assignment_mode),
            status == cp_model.OPTIMAL,
        )
    if in_main_solve:
        print("unfeasible solution in ortools")
    return -1, -1, None, None


def ComputeDelaysBetweenNodes(problem, task_intervals):
    """Computes the min delays between all pairs of tasks in 'task_intervals'.
    Args:
      problem: The protobuf of the model.
      task_intervals: The output of the AnalysePrecedenceGraph().
    Returns:
      a list of (task1, task2, min_delay_between_task1_and_task2)
    """
    # print("Computing the minimum delay between pairs of intervals")
    delays = {}

    complete_problem_assignment = None
    num_optimal_delays = 0
    num_delays_not_found = 0
    optimal_found = True
    for start_task, end_task, active_tasks in task_intervals:
        min_delay, feasible_delay, assignment, _ = SolveRcpsp(
            problem,
            "",
            # "num_search_workers:16,max_time_in_seconds:0.0",
            "",
            active_tasks,
            start_task,
            end_task,
            [],
            delays,
        )
        if min_delay != -1:
            delays[(start_task, end_task)] = min_delay, feasible_delay
            if start_task == 0 and end_task == len(problem.tasks) - 1:
                complete_problem_assignment = assignment
            if min_delay == feasible_delay:
                num_optimal_delays += 1
            else:
                optimal_found = False
        else:
            num_delays_not_found += 1
            optimal_found = False

    # print(f"  - #optimal delays = {num_optimal_delays}", flush=True)
    # if num_delays_not_found:
    # print(f"  - #not computed delays = {num_delays_not_found}", flush=True)

    return delays, complete_problem_assignment, optimal_found
