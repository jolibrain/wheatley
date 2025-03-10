import os

import random
import sys
from typing import Tuple

import numpy as np
import pytest
import torch
import shutil

from alg.ppo import PPO
from alg.pretrain import Pretrainer
from args import argument_parser, parse_args
from generic.utils import get_path
from generic.agent_specification import AgentSpecification
from generic.agent_validator import AgentValidator
from generic.training_specification import TrainingSpecification
from psp.description import Description
from psp.env.env import Env
from psp.env.env_specification import EnvSpecification
from psp.models.agent import Agent
from psp.train_psp import main
from psp.utils.loaders import PSPLoader


def instantiate_training_objects(
    args: list,
) -> Tuple[
    PPO, Pretrainer, AgentValidator, EnvSpecification, Description, AgentSpecification
]:
    original_argv = sys.argv

    try:
        # Simulate the arguments passed as input for the argument parser to work seemlessly.
        args.append("--disable_visdom")
        if len(original_argv) == 3:
            output_directory = original_argv[2]
            output_directory += "/" if not output_directory.endswith("/") else ""
            args.append("--path")
            args.append(output_directory)

        sys.argv = ["python3"] + args

        parser = argument_parser()
        args, exp_name = parse_args(parser)
    finally:
        # Don't forget to bring back the old argv!
        sys.argv = original_argv

    exp_name = args.exp_name_appendix
    path = get_path(args.path, exp_name)

    loader = PSPLoader(args.generate_duration_bounds)

    if args.load_problem is None:
        if args.train_dir is None:
            raise RuntimeError("--train_dir is mandatory")
        else:
            train_psps = loader.load_directory(args.train_dir)

        if args.test_dir is not None:
            test_psps = loader.load_directory(args.test_dir)
        else:
            test_psps = []
            for i in range(int(len(train_psps) * args.train_test_split)):
                tp = random.choice(train_psps)
                test_psps.append(tp)
                train_psps.remove(tp)
    else:
        psp = loader.load_single(args.load_problem)
        train_psps = [psp]
        test_psps = [psp]

    problem_description = Description(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        train_psps=train_psps,
        test_psps=test_psps,
        seed=args.seed,
    )
    training_specification = TrainingSpecification(
        total_timesteps=args.total_timesteps,
        n_validation_env=len(test_psps),
        fixed_validation=args.fixed_validation,
        fixed_random_validation=args.fixed_random_validation,
        validation_batch_size=args.validation_batch_size,
        validation_freq=1 if args.validation_freq == -1 else args.validation_freq,
        display_env=exp_name,
        path=path,
        custom_heuristic_names=args.custom_heuristic_names,
        ortools_strategy=args.ortools_strategy,
        max_time_ortools=args.max_time_ortools,
        scaling_constant_ortools=args.scaling_constant_ortools,
        vecenv_type=args.vecenv_type,
        validate_on_total_data=args.validate_on_total_data,
        optimizer=args.optimizer,
        n_workers=args.n_workers,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        normalize_advantage=not args.dont_normalize_advantage,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_steps_episode=args.n_steps_episode,
        batch_size=args.batch_size,
        iter_size=args.iter_size,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        freeze_graph=args.freeze_graph,
        lr=args.lr,
        fe_lr=args.fe_lr,
        rpo=args.rpo,
        rpo_smoothing_param=args.rpo_smoothing_param,
        gae_lambda=args.gae_lambda,
        return_based_scaling=args.return_based_scaling,
        store_rollouts_on_disk=(
            args.store_rollouts_on_disk if args.vecenv_type == "graphgym" else None
        ),
        critic_loss=args.critic_loss,
        debug_net=False,
        display_gantt=False,
        max_shared_mem_per_worker=args.max_shared_mem_per_worker,
        espo=False,
    )

    if args.conflicts == "clique" and args.precompute_cliques:
        observe_clique = True
    else:
        observe_clique = False
    if args.observe_duration_when_affect:
        observe_real_duration_when_affect = True
    else:
        observe_real_duration_when_affect = False
    env_specification = EnvSpecification(
        problems=problem_description,
        normalize_input=not args.dont_normalize_input,
        input_list=args.features,
        max_edges_factor=args.max_edges_upper_bound_factor,
        sample_n_jobs=args.sample_n_jobs,
        chunk_n_jobs=args.chunk_n_jobs,
        observe_conflicts_as_cliques=observe_clique,
        add_rp_edges=args.add_rp_edges,
        observe_real_duration_when_affect=observe_real_duration_when_affect,
        do_not_observe_updated_bounds=args.do_not_observe_updated_bounds,
        factored_rp=(args.fe_type == "tokengt" or args.factored_rp),
        remove_old_resource_info=args.remove_old_resource_info
        and not args.observe_subgraph,
        remove_past_prec=not args.keep_past_prec and not args.observe_subgraph,
        observation_horizon_step=args.observation_horizon_step,
        observation_horizon_time=args.observation_horizon_time,
        fast_forward=args.fast_forward,
        observe_subgraph=args.observe_subgraph,
        random_taillard=args.random_taillard,
        pyg=True,
    )

    if args.batch_size == 1 and not args.dont_normalize_advantage:
        raise RuntimeError(
            "batch size 1 and normalize advantage are not compatible\neither set --batch_size to > 1  or append --dont_normalize_advantage"
        )

    agent_specification = AgentSpecification(
        n_features=env_specification.n_features,
        gconv_type=args.gconv_type,
        graph_has_relu=args.graph_has_relu,
        graph_pooling=args.graph_pooling,
        layer_pooling=args.layer_pooling if not args.hierarchical else "last",
        mlp_act=args.mlp_act,
        mlp_act_graph=args.mlp_act_graph,
        device=torch.device(args.device),
        n_mlp_layers_features_extractor=args.n_mlp_layers_features_extractor,
        n_layers_features_extractor=args.n_layers_features_extractor,
        hidden_dim_features_extractor=args.hidden_dim_features_extractor,
        n_attention_heads=args.n_attention_heads,
        reverse_adj=args.reverse_adj_in_gnn,
        residual_gnn=args.residual_gnn,
        normalize_gnn=args.normalize_gnn,
        conflicts=args.conflicts,
        n_mlp_layers_actor=args.n_mlp_layers_actor,
        hidden_dim_actor=args.hidden_dim_actor,
        n_mlp_layers_critic=args.n_mlp_layers_critic,
        hidden_dim_critic=args.hidden_dim_critic,
        fe_type=args.fe_type,
        transformer_flavor=args.transformer_flavor,
        dropout=args.dropout,
        cache_lap_node_id=not args.dont_cache_lap_node_id,
        lap_node_id_k=args.lap_node_id_k,
        edge_embedding_flavor=args.edge_embedding_flavor,
        performer_nb_features=args.performer_nb_features,
        performer_feature_redraw_interval=args.performer_redraw_interval,
        performer_generalized_attention=args.performer_generalized_attention,
        performer_auto_check_redraw=args.performer_auto_check_redraw,
        vnode=args.vnode,
        update_edge_features=args.update_edge_features,
        update_edge_features_pe=args.update_edge_features_pe,
        ortho_embed=args.ortho_embed,
        no_tct=args.no_tct,
        mid_in_edges=args.mid_in_edges,
        rwpe_k=args.rwpe_k,
        rwpe_h=args.rwpe_h,
        cache_rwpe=args.cache_rwpe,
        two_hot=args.two_hot,
        symlog=args.symlog,
        hl_gauss=args.hl_gauss,
        reward_weights=args.reward_weights,
        sgformer=args.sgformer,
        pyg=True,
        hierarchical=args.hierarchical,
        dual_net=False,
    )
    agent = Agent(
        env_specification=env_specification, agent_specification=agent_specification
    )

    agent.to(args.device)
    pretrainer = Pretrainer(
        problem_description,
        env_specification,
        training_specification,
        Env,
        num_envs=args.pretrain_num_envs,
        prob=args.pretrain_prob,
    )
    validator = AgentValidator(
        problem_description,
        env_specification,
        args.device,
        training_specification,
        args.disable_visdom,
    )
    ppo = PPO(
        training_specification,
        Env,
        validator,
    )

    return (
        ppo,
        pretrainer,
        validator,
        env_specification,
        problem_description,
        agent_specification,
    )


# This is the list of all experiments we want to try.
# Each entry should be read as "argument_name: [value_experiment_1, value_experiment_2, ...]"
# If an entry has less experiment values than the maximum number of experiments,
# then its last value will be repeated for the missing experiment values.
possible_args = {
    "n_j": [6, 10],
    "n_m": [10, 5],
    "dont_normalize_input": [True, False],
    # "sample_n_jobs": [-1, 3],  # WARNING: This option does not work for PSPs?
    # "chunk_n_jobs": [3, -1],
    "max_n_j": [20],
    "max_n_m": [20],
    "lr": [0.0001],
    "ent_coef": [0.05],
    "vf_coef": [1.0],
    "clip_range": [0.25],
    "gamma": [1.0],
    "gae_lambda": [1.0],
    "optimizer": ["radam"],
    "fe_type": ["message_passing"],
    "residual_gnn": [False, True],
    "graph_has_relu": [True],
    "graph_pooling": ["learn", "max"],
    "hidden_dim_features_extractor": [32],
    "n_layers_features_extractor": [3],
    "mlp_act": ["gelu"],
    "layer_pooling": ["all", "last"],
    "n_mlp_layers_features_extractor": [1],
    "n_mlp_layers_actor": [1],
    "n_mlp_layers_critic": [1],
    "hidden_dim_actor": [16],
    "hidden_dim_critic": [16],
    "total_timesteps": [1000],
    "n_validation_env": [10],
    "n_steps_episode": [490],
    "batch_size": [245],
    "n_epochs": [1],
    "fixed_validation": [True],
    # "custom_heuristic_names": ["SPT MWKR MOPNR FDD/MWKR", "SPT"],
    "seed": [0],
    "duration_type": ["stochastic", "deterministic"],
    "generate_duration_bounds": ["0.05 0.1"],
    "ortools_strategy": ["averagistic", "realistic", "pessimistic"],
    "device": ["cuda:0"],
    "n_workers": [2],
    "skip_initial_eval": [True, False],
    "exp_name_appendix": ["test"],
    "train_dir": ["./instances/psp/test/"],
    "vecenv_type": ["graphgym"],
    # "return_based_scaling": [True, False],
    "observe_subgraph": [False],
    "fast_forward": [True, False],
    "observation_horizon_step": [2, 5],
    "observation_horizon_time": [2, 5],
    "symlog": [True, False],
    "store_rollouts_on_disk": [False, "/tmp/"],
    "critic_loss": ["l2", "l1"],
    "hierarchical": [False],
}

# Duplicate each entry to match the maximum number of possibilities to try.
max_number_of_possibilities = max(len(v) for v in possible_args.values())
possible_args = {
    k: v + (max_number_of_possibilities - len(v)) * [v[-1]]
    for k, v in possible_args.items()
}

# Build the list of all experiments to launch.
# An experiment is defined by a list of arguments preformatted
# for argparse.
args_to_test = [[] for _ in range(max_number_of_possibilities)]
for key, values in possible_args.items():
    for test_id, value in enumerate(values):
        args = args_to_test[test_id]

        if value is True:
            args.append(f"--{key}")
        elif value is False:
            pass
        else:
            args.append(f"--{key}")
            if isinstance(value, str):
                for sub_v in value.split(" "):
                    args.append(f"{sub_v}")
                    print(sub_v)
            else:
                args.append(f"{value}")


@pytest.mark.parametrize(
    "args",
    args_to_test,
)
def test_args(args: list):
    """Make sure the main training function don't crash when using multiple different
    args.
    """
    original_argv = sys.argv
    print("ARGS", args)

    try:
        # Simulate the arguments passed as input for the argument parser to work seemlessly.
        args.append("--disable_visdom")
        if len(original_argv) == 3:
            output_directory = original_argv[2]
            output_directory += "/" if not output_directory.endswith("/") else ""
            args.append("--path")
            args.append(output_directory)

        sys.argv = ["python3"] + args

        parser = argument_parser()
        args, exp_name = parse_args(parser)
        main(args, exp_name)
    finally:
        # Don't forget to bring back the old argv!
        sys.argv = original_argv
        shutil.rmtree("./saved_networks/test")


# @pytest.mark.parametrize(
#     "args,problem_1,problem_2",
#     [
#         (
#             [
#                 "--exp_name_appendix",
#                 "test-save-ortools",
#                 "--n_steps_episode",
#                 "64",
#                 "--n_workers",
#                 "1",
#                 "--batch_size",
#                 "4",
#                 "--n_epochs",
#                 "1",
#                 "--device",
#                 "cuda:0",
#                 "--total_timesteps",
#                 "64",
#                 "--n_layers_features_extractor",
#                 "8",
#                 "--residual_gnn",
#                 "--clip_range",
#                 "0.25",
#                 "--target_kl",
#                 "0.2",
#                 "--gae_lambda",
#                 "0.95",
#                 "--hidden_dim_features_extractor",
#                 "64",
#                 "--vecenv_type",
#                 "graphgym",
#             ],
#             "./instances/psp/small/small.sm",
#             "./instances/psp/sm/j60/j6010_1.sm",
#         )
#     ],
# )
# def test_save_ortools(args: list, problem_1: str, problem_2: str):
#     """Make sure the main training function don't crash when using multiple different
#     args.
#     """
#     args_1 = args + ["--load_problem", problem_1]
#     _, _, agent_validator, _, _, _ = instantiate_training_objects(args_1)
#     file_path = agent_validator._ortools_solution_path(0, "realistic")

#     criterion, schedule, optimal = agent_validator._get_ortools_criterion(
#         0, "realistic", "makespan"
#     )

#     assert (
#         agent_validator._ortools_read_solution(
#             file_path, agent_validator.validation_envs[0].problem
#         )
#         is not None
#     ), "Solution should be found"

#     (
#         saved_criterion,
#         saved_schedule,
#         saved_optimal,
#     ) = agent_validator._ortools_read_solution(
#         file_path, agent_validator.validation_envs[0].problem
#     )

#     assert criterion == saved_criterion, "Not the same criterion."
#     assert optimal == saved_optimal, "Not the same optimal."
#     for i in range(len(schedule)):
#         assert np.all(schedule[i] == saved_schedule[i]), f"Not the same schedule ({i})"

#     args_2 = args + ["--load_problem", problem_2]
#     _, _, agent_validator, _, _, _ = instantiate_training_objects(args_2)
#     file_path = agent_validator._ortools_solution_path(0, "realistic")

#     assert os.path.exists(file_path), "Saving file should exists."
#     assert (
#         agent_validator._ortools_read_solution(
#             file_path, agent_validator.validation_envs[0].problem
#         )
#         is None
#     ), "Solution should not be found"
