# Running benchmark on Taillard's problems and storing the results in taillard_results.txt.
# If you want to run the benchmark with default parameters on all available taillards problems (stored in
# wheatley/instances/taillard/), just run: python3 run_taillard.py


import glob
import os
import sys

import numpy as np
import torch

sys.path.append("..")

from args import args  # noqa E402
from env.env_specification import EnvSpecification  # noqa E402
from models.agent import Agent  # noqa E402
from models.agent_specification import AgentSpecification  # noqa E402
from models.training_specification import TrainingSpecification  # noqa E402
from problem.problem_description import ProblemDescription  # noqa E402
from utils.ortools_solver import solve_jssp  # noqa E402
from utils.utils import get_exp_name, get_n_features, get_path, load_taillard_problem  # noqa E402


def main():

    # Get Taillard problem files
    problem_files = glob.glob("../instances/taillard/" + args.taillard_pbs + ".txt")

    # Iterate problems
    taillards = {}
    for problem_file in problem_files:
        # Load problem
        taillard_problem = os.path.basename(problem_file).replace(".txt", "")
        n_j, n_m, affectations, durations = load_taillard_problem(problem_file)

        # Update arguments
        args.n_j = n_j
        args.n_m = n_m
        args.max_n_j = n_j
        args.max_n_m = n_m
        args.n_steps_episode = n_j * n_m * 5
        args.batch_size = n_j * n_m
        args.total_timesteps = 10000 * n_j * n_m
        args.n_validation_env = 1
        args.fixed_problem = True
        exp_name = get_exp_name(args)
        path = get_path(args.path, exp_name)

        print("Training the agent")

        # Define problem
        problem_description = ProblemDescription(
            transition_model_config=args.transition_model_config,
            reward_model_config=args.reward_model_config,
            deterministic=True,
            fixed=True,
            affectations=affectations,
            durations=durations,
        )
        problem_description.print_self()
        training_specification = TrainingSpecification(
            total_timesteps=args.total_timesteps,
            n_validation_env=args.n_validation_env,
            validation_freq=args.validation_freq,
            display_env=exp_name,
            path=path,
            custom_heuristic_name=args.custom_heuristic_name,
            ortools_strategy=args.ortools_strategy,
            max_time_ortools=args.max_time_ortools,
            scaling_constant_ortools=args.scaling_constant_ortools,
        )
        training_specification.print_self()
        env_specification = EnvSpecification(
            max_n_jobs=args.max_n_j,
            max_n_machines=args.max_n_m,
            normalize_input=not args.dont_normalize_input,
            input_list=args.features,
            insertion_mode=args.insertion_mode,
        )
        env_specification.print_self()
        agent_specification = AgentSpecification(
            lr=args.lr,
            n_steps_episode=args.n_steps_episode,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            optimizer=args.optimizer,
            freeze_graph=args.freeze_graph,
            n_features=get_n_features(args.features, args.max_n_j, args.max_n_m),
            gconv_type=args.gconv_type,
            graph_has_relu=args.graph_has_relu,
            graph_pooling=args.graph_pooling,
            mlp_act=args.mlp_act,
            n_workers=args.n_workers,
            device=torch.device(args.device),
            n_mlp_layers_features_extractor=args.n_mlp_layers_features_extractor,
            n_layers_features_extractor=args.n_layers_features_extractor,
            hidden_dim_features_extractor=args.hidden_dim_features_extractor,
            n_attention_heads=args.n_attention_heads,
            n_mlp_layers_actor=args.n_mlp_layers_actor,
            hidden_dim_actor=args.hidden_dim_actor,
            n_mlp_layers_critic=args.n_mlp_layers_critic,
            hidden_dim_critic=args.hidden_dim_critic,
        )
        agent_specification.print_self()
        agent = Agent(env_specification=env_specification, agent_specification=agent_specification)

        # Launch training
        agent.train(problem_description, training_specification)

        print("Testing the agent")

        # We first load the agent, since the last agent object is not necessarly the best one
        agent = Agent.load(path)
        rl_makespan = agent.predict(problem_description).get_makespan()
        ortools_makespan = solve_jssp(
            affectations, durations, args.max_time_ortools, args.scaling_constant_ortools
        ).get_makespan()

        # Store the best metric
        print("Taillard ", taillard_problem, " / best makespace=", rl_makespan)
        taillards[taillard_problem] = rl_makespan
        with open("taillard_results.txt", "a") as result_file:
            result_file.write(f"{taillard_problem}. RL Score={rl_makespan}. OR-Tools Score={ortools_makespan}")

    # Print the results
    print("Taillard benchmark results")
    print(taillards)


if __name__ == "__main__":
    main()
