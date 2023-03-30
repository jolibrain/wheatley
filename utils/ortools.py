#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

import time

import matplotlib.pyplot as plt
import numpy as np
import visdom
import collections

from env.jssp_state import JSSPState as State
from problem.jssp_description import JSSPDescription as ProblemDescription
from utils.utils import obs_as_tensor_add_batch_dim, decode_mask
from ortools.sat.python import cp_model
from problem.solution import Solution

import torch


def solve_jssp(
    affectations, durations, n_features, max_time_ortools, scaling_constant_ortools
):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = []
    for i in range(affectations.shape[0]):
        jobs_data.append([])
        for j in range(affectations.shape[1]):
            if affectations[i, j] != -1:
                jobs_data[-1].append(
                    (
                        int(affectations[i, j]),
                        int(float(durations[i, j]) * scaling_constant_ortools),
                    )
                )

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
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_ortools
    status = solver.Solve(model)

    schedule = np.zeros_like(affectations)

    # Create one list of assigned tasks per machine.
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            schedule[job_id, task_id] = solver.Value(all_tasks[job_id, task_id].start)
    return (
        Solution(
            schedule=schedule / scaling_constant_ortools, real_durations=durations
        ),
        status == cp_model.OPTIMAL,
    )


def get_ortools_makespan(
    affectations,
    durations,
    n_features,
    max_time_ortools,
    scaling_constant_ortools,
    ortools_strategy="pessimistic",
):
    n_j, n_m = affectations.shape[0], affectations.shape[1]

    if ortools_strategy == "realistic" or durations.shape[2] == 1:
        durs = durations[:, :, 0]
    elif ortools_strategy == "pessimistic":
        durs = durations[:, :, 2]
    elif ortools_strategy == "optimistic":
        durs = durations[:, :, 1]
    elif ortools_strategy == "averagistic":
        durs = durations[:, :, 3]
    else:
        print("unknow ortools strategy ", ortools_strategy)
        exit()

    solution, optimal = solve_jssp(
        affectations, durs, n_features, max_time_ortools, scaling_constant_ortools
    )

    if durations.shape[2] == 1:
        return solution.get_makespan(), solution.schedule

    state = State(
        affectations,
        durations,
        affectations.shape[0],
        affectations.shape[1],
        n_features,
    )
    state.reset()

    # use the same durations to compute machines occupancies
    state.durations = state.original_durations
    for i in range(n_j * n_m):
        state.affect_node(i)
    # we will use get_machine_occupancy_max_endtime which rely on task_completion_times
    state.set_all_task_completion_times(
        np.expand_dims(solution.schedule, 2) + durations
    )

    occupancies = [state.get_machine_occupancy(m, ortools_strategy) for m in range(n_m)]

    state.reset()  # reset task_completion times in particular

    # observe all real durations
    for i in range(n_j * n_m):
        state.observe_real_duration(i, do_update=False)
        state.affect_node(i)

    for o in occupancies:
        nodes_occ = [el[2] for el in o]
        for i in range(len(nodes_occ) - 1):
            state.set_precedency(nodes_occ[i], nodes_occ[i + 1], do_update=False)

    state.update_completion_times_from_sinks()

    tct = (
        state.get_all_task_completion_times()[:, 0]
        .reshape(n_j, n_m, 1)
        .squeeze_(2)
        .numpy()
    )

    makespan = torch.max(state.get_all_task_completion_times()[:, 0].flatten())

    return makespan, tct - durations[:, :, 0], optimal


def get_ortools_schedule(
    env,
    max_time_ortools=10,
    scaling_constant_ortools=1000,
    ortools_strategy="averagistic",
):
    _, ortools_schedule, optimal = get_ortools_makespan(
        env.state.affectations,
        env.state.original_durations,
        env.env_specification.n_features,
        max_time_ortools,
        scaling_constant_ortools,
        ortools_strategy,
    )
    return ortools_schedule, optimal


def get_ortools_actions(env, ortools_schedule):
    nodes_time = ortools_schedule.flatten()
    nodes_machine = env.state.affectations.flatten()
    nodes_affected = env.state.affected.flatten()

    # for each machine, get the first node time to be scheduled
    machine_next = {}
    for node_id, machine_id in enumerate(nodes_machine):
        if machine_id == -1:
            continue
        if nodes_affected[node_id]:
            continue
        previous = machine_next.get(machine_id, float("inf"))
        current = nodes_time[node_id]
        if current < previous:
            machine_next[machine_id] = current

    # for each valid action, check if it is the minimum node time for this machine
    mask = env.action_masks()
    valid_actions = [node for node, masked in enumerate(mask) if masked == True]
    ortools_actions = []
    for action in valid_actions:
        machine_id = nodes_machine[action]
        if nodes_time[action] != machine_next[machine_id]:
            continue
        ortools_actions.append(action)

    return ortools_actions


def get_ortools_trajectory_and_past_actions(env):
    schedule, optimal = get_ortools_schedule(env)
    # if optimal:
    #     print("using optimal solution from or-tools")
    trajectory = []
    past_actions = []
    obs = []
    masks = []
    o, info = env.reset()
    next_obs = obs_as_tensor_add_batch_dim(o)
    action_mask = decode_mask(info["mask"])
    while not env.done():
        obs.append(next_obs)
        masks.append(action_mask)
        actions = get_ortools_actions(env, schedule)
        action = np.random.choice(actions)
        # action = actions[0]
        next_obs, reward, done, _, info = env.step(action)
        action_mask = decode_mask(info["mask"])
        next_obs = obs_as_tensor_add_batch_dim(next_obs)
        past_actions.append(trajectory.copy())
        trajectory.append(action)
    return obs, masks, trajectory, past_actions
