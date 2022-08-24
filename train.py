import os

import numpy as np
import torch

from env.env_specification import EnvSpecification
from models.agent import Agent
from models.agent_specification import AgentSpecification
from models.training_specification import TrainingSpecification
from problem.problem_description import ProblemDescription
from utils.utils import get_n_features, generate_deterministic_problem, generate_problem_distrib, load_problem

from args import args, exp_name, path


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # If we want to load a specific problem, under the taillard (extended) format, and train on it, we first do it.
    # Note that this problem can be stochastic or deterministic
    affectations, durations = None, None
    if args.load_problem is not None:
        args.n_j, args.n_m, affectations, durations = load_problem(
            args.load_problem,
            taillard_offset=False,
            deterministic=(args.duration_type == "deterministic"),
            load_from_job=args.load_from_job,
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
    problem_description.print_self()

    # Then specify the variables used for the training
    training_specification = TrainingSpecification(
        total_timesteps=args.total_timesteps,
        n_validation_env=args.n_validation_env,
        fixed_validation=args.fixed_validation,
        fixed_random_validation=args.fixed_random_validation,
        validation_batch_size=args.validation_batch_size,
        validation_freq=args.n_steps_episode * args.n_workers if args.validation_freq == -1 else args.validation_freq,
        display_env=exp_name,
        path=path,
        custom_heuristic_name=args.custom_heuristic_name,
        ortools_strategy=args.ortools_strategy,
        max_time_ortools=args.max_time_ortools,
        scaling_constant_ortools=args.scaling_constant_ortools,
        vecenv_type=args.vecenv_type,
    )
    training_specification.print_self()

    # If we want to use a pretrained Agent, we only have to load it (if it exists)
    if args.retrain and os.path.exists(path + ".zip"):
        print("Retraining an already existing agent\n")
        agent = Agent.load(path)

    # Else, we instantiate a new Agent
    else:
        env_specification = EnvSpecification(
            max_n_jobs=args.max_n_j,
            max_n_machines=args.max_n_m,
            normalize_input=not args.dont_normalize_input,
            input_list=args.features,
            insertion_mode=args.insertion_mode,
            max_edges_factor=args.max_edges_upper_bound_factor,
            sample_n_jobs=args.sample_n_jobs,
            chunk_n_jobs=args.chunk_n_jobs,
        )
        env_specification.print_self()
        agent_specification = AgentSpecification(
            lr=args.lr,
            fe_lr=args.fe_lr,
            n_steps_episode=args.n_steps_episode,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            normalize_advantage=not args.dont_normalize_advantage,
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
            reverse_adj=args.reverse_adj_in_gnn,
            residual_gnn=args.residual_gnn,
            normalize_gnn=args.normalize_gnn,
            conflicts_edges=args.conflicts_edges,
            n_mlp_layers_shared=args.n_mlp_layers_shared,
            hidden_dim_shared=args.hidden_dim_shared,
            n_mlp_layers_actor=args.n_mlp_layers_actor,
            hidden_dim_actor=args.hidden_dim_actor,
            n_mlp_layers_critic=args.n_mlp_layers_critic,
            hidden_dim_critic=args.hidden_dim_critic,
        )
        agent_specification.print_self()
        agent = Agent(env_specification=env_specification, agent_specification=agent_specification)

    # And finally, we train the model on the specified training mode
    # Note: The saving of the best model is hanlded in the agent.train method.
    # We save every time we hit a min RL / OR-Tools ratio
    agent.train(problem_description, training_specification)


if __name__ == "__main__":
    main()
