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

import numpy as np
import torch

from env.env_specification import EnvSpecification
from models.agent import Agent
from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.utils_testing import get_ortools_makespan

from args import args, exp_name, path


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading agent")
    agent = Agent.load(path)
    random_agent = RandomAgent(args.max_n_j, args.max_n_m)

    print(
        "Launching inference.\n"
        f"Problem size : {args.n_j} jobs, {args.n_m} machines\n"
        f"Number of problem tested : {args.n_test_problems}"
    )

    diff_percentages = []
    rl_makespans = []
    or_tools_makespans = []
    random_makespans = []

    problem_description = ProblemDescription(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        fixed=args.fixed_problem,
        n_jobs=args.n_j,
        n_machines=args.n_m,
        max_duration=args.max_duration,
    )
    env_specification = EnvSpecification(
        max_n_jobs=args.max_n_j,
        max_n_machines=args.max_n_m,
        normalize_input=not args.dont_normalize_input,
        input_list=args.features,
        insertion_mode=args.insertion_mode,
    )

    for i in range(args.n_test_problems):
        # Prints
        if (i + 1) % (args.n_test_problems // args.test_print_every) == 0:
            print(f"{i+1}/{args.n_test_problems}")

        # We use a frozen version of our problem_description, to have the same problem for RL, Random, and OR-Tools
        frozen_problem_description = problem_description.get_frozen_version()
        affectations, durations = frozen_problem_description.sample_problem()

        rl_makespan = agent.predict(frozen_problem_description).get_makespan()
        random_makespan = random_agent.predict(frozen_problem_description, env_specification).get_makespan()
        or_tools_makespan = get_ortools_makespan(
            affectations,
            durations,
            args.max_time_ortools,
            args.scaling_constant_ortools,
        )[0]

        diff_percentage = 100 * (rl_makespan - or_tools_makespan) / or_tools_makespan
        rl_makespans.append(rl_makespan)
        or_tools_makespans.append(or_tools_makespan)
        random_makespans.append(random_makespan)
        diff_percentages.append(diff_percentage)

    print(f"Makespan for RL solution : {np.mean(rl_makespans):.0f}±{np.std(rl_makespans):.0f}")
    print(f"Makespan for OR-tools solution : {np.mean(or_tools_makespans):.0f}±{np.std(or_tools_makespans):.0f}")
    print(f"Makespan for random solution : {np.mean(random_makespans):.0f}±{np.std(random_makespans):.0f}")
    print(
        "Difference in percentage between OR-tools and RL : "
        f"{np.mean(diff_percentages):.1f}±{np.std(diff_percentages):.1f}%"
    )


if __name__ == "__main__":
    main()
