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

import sys

sys.path.append("..")

import csv
import itertools
import json
import os
import socket
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from args import args, exp_name, path
from env.env import Env
from env.env_specification import EnvSpecification
from models.agent import Agent
from models.agent_specification import AgentSpecification
from models.training_specification import TrainingSpecification
from problem.problem_description import ProblemDescription
from utils.ortools import get_ortools_makespan
from utils.utils import (
    generate_deterministic_problem,
    generate_problem_distrib,
    get_n_features,
    load_problem,
)

EPSILON = 1e-3  # because of OR-tools int/float conversion, ignore small overlaps


# count how many tasks overlap if there all start as soon as possible
def metric_overlap(state):
    total = 0
    # check all pairs of jobs
    for j1, j2 in itertools.combinations(range(state.n_jobs), 2):
        # check all pairs of machine
        for m1 in range(state.n_machines):
            # not affected
            if state.affectations[j1][m1] == -1:
                continue
            for m2 in range(m1 + 1, state.n_machines):
                # not the same machine
                if state.affectations[j1][m1] != state.affectations[j2][m2]:
                    continue
                # no overlap
                tct = state.get_all_task_completion_times()
                stop1 = tct[j1 * state.n_machines + m1][3]
                start1 = stop1 - state.durations[j1][m1][3]
                stop2 = tct[j2 * state.n_machines + m2][3]
                start2 = stop2 - state.durations[j2][m2][3]
                if min(stop1, stop2) - max(start1, start2) <= 0:
                    continue
                total += 1
                # print("conflict: [{:.2f} {:.2f}] and [{:.2f} {:.2f}] on machine {} {}".format(start1, stop1, start2, stop2, state.affectations[j1][m1], state.affectations[j2][m2]))
    return total / state.n_jobs


# maximum of tasks scheduled in //
def metric_parallel(state, schedule):
    intervals = []
    times = []
    all_finish = schedule + state.original_durations[:, :, 0]
    for j in range(state.n_jobs):
        for m in range(state.n_machines):
            if state.affectations[j][m] == -1:
                continue
            start = schedule[j][m]
            finish = all_finish[j][m]
            assert start < finish
            intervals.append([start, finish])
            times.append(start)
    maxi = 0
    for time in times:
        current = 0
        for start, finish in intervals:
            if start <= time < finish - EPSILON:
                # print(time, 'in interval [', start, '-', finish - EPSILON, ']')
                current += 1
        maxi = max(maxi, current)
    return maxi


# % of waiting time per job (except at the end)
def metric_gap(state, makespan, schedule):
    all_finish = schedule + state.original_durations[:, :, 0]
    total = 0
    for j in range(state.n_jobs):
        intervals = []
        for m in range(state.n_machines):
            if state.affectations[j][m] == -1:
                continue
            start = schedule[j][m]
            finish = all_finish[j][m]
            assert start < finish
            intervals.append([start, finish])
        intervals.sort()
        total += intervals[0][0]  # waiting time at the beginning
        for i in range(len(intervals) - 1):
            stop = intervals[i][1]
            start = intervals[i + 1][0]
            gap = start - stop
            assert gap >= -EPSILON  # accept small errors
            gap = abs(gap)
            if gap < EPSILON:
                gap = 0
            total += gap
    return 100 * total / makespan / state.n_jobs


def sampling(sample_n_jobs, writer=None):
    args.sample_n_jobs = args.max_n_j = sample_n_jobs

    # If we want to load a specific problem, under the taillard (extended) format, and train on it, we first do it.
    # Note that this problem can be stochastic or deterministic
    affectations, durations = None, None
    if args.load_problem is not None:
        args.n_j, args.n_m, affectations, durations = load_problem(
            args.load_problem,
            taillard_offset=False,
            deterministic=(args.duration_type == "deterministic"),
            load_max_jobs=args.load_max_jobs,
            generate_bounds=args.generate_duration_bounds,
        )
        args.fixed_problem = True

    # Define problem and visualize it
    problem_description = ProblemDescription(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        fixed=args.fixed_problem,
        affectations=affectations,
        durations=durations,
        n_jobs=args.n_j,
        n_machines=args.n_m,
        max_duration=args.max_duration,
        duration_mode_bounds=args.duration_mode_bounds,
        duration_delta=args.duration_delta,
    )

    # Then specify the variables used for the training
    training_specification = TrainingSpecification(
        total_timesteps=args.total_timesteps,
        n_validation_env=args.n_validation_env,
        validation_freq=args.n_steps_episode * args.n_workers
        if args.validation_freq == -1
        else args.validation_freq,
        display_env=exp_name,
        path=path,
        custom_heuristic_names=args.custom_heuristic_names,
        ortools_strategy=args.ortools_strategy,
        max_time_ortools=args.max_time_ortools,
        scaling_constant_ortools=args.scaling_constant_ortools,
        vecenv_type=args.vecenv_type,
    )

    # Instantiate a new Agent
    env_specification = EnvSpecification(
        max_n_jobs=args.max_n_j,
        max_n_machines=args.max_n_m,
        normalize_input=not args.dont_normalize_input,
        input_list=args.features,
        insertion_mode=args.insertion_mode,
        max_edges_factor=args.max_edges_upper_bound_factor,
        sample_n_jobs=args.sample_n_jobs,
    )

    env = Env(problem_description, env_specification)
    overlaps = []
    parallels = []
    gaps = []
    for i in range(100):
        env.reset()
        or_tools_makespan, or_tools_schedule = get_ortools_makespan(
            env.state.affectations,
            env.state.original_durations,
            training_specification.max_time_ortools,
            training_specification.scaling_constant_ortools,
            training_specification.ortools_strategy,
        )
        overlaps.append(metric_overlap(env.state))
        parallels.append(metric_parallel(env.state, or_tools_schedule))
        gaps.append(metric_gap(env.state, or_tools_makespan.item(), or_tools_schedule))
    assert max(parallels) <= sample_n_jobs
    writer.writerow(
        [
            sample_n_jobs,
            sum(overlaps) / len(overlaps),
            sum(parallels) / len(parallels),
            sum(gaps) / len(gaps),
        ]
    )


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    f = open("sampling.csv", "w")
    writer = csv.writer(f)
    writer.writerow(
        [
            "sample_n_jobs",
            "overlaps",
            "parallels",
            "gaps",
            socket.gethostname(),
            " ".join(sys.argv),
        ]
    )

    benchs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for i in benchs:
        sampling(i, writer)


if __name__ == "__main__":
    main()
