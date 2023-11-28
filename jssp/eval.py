"""Evaluate trained model on all instances size."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from generic.agent_specification import AgentSpecification
from generic.agent_validator import AgentValidator
from generic.training_specification import TrainingSpecification
from jssp.description import Description
from jssp.env.env import Env
from jssp.env.env_specification import EnvSpecification
from jssp.env.state import State
from jssp.models.agent import Agent


def load_agent(args: argparse.Namespace, path: str) -> Agent:
    env_specification = EnvSpecification(
        max_n_jobs=args.max_n_j,
        max_n_machines=args.max_n_m,
        normalize_input=not args.dont_normalize_input,
        input_list=args.features,
        insertion_mode=args.insertion_mode,
        max_edges_factor=args.max_edges_upper_bound_factor,
        sample_n_jobs=args.sample_n_jobs,
        chunk_n_jobs=args.chunk_n_jobs,
        observe_conflicts_as_cliques=args.conflicts == "clique"
        and args.precompute_cliques,
        observe_real_duration_when_affect=args.observe_duration_when_affect,
        do_not_observe_updated_bounds=args.do_not_observe_updated_bounds,
    )
    agent_specification = AgentSpecification(
        n_features=env_specification.n_features,
        gconv_type=args.gconv_type,
        graph_has_relu=args.graph_has_relu,
        graph_pooling=args.graph_pooling,
        layer_pooling=args.layer_pooling,
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
        update_edge_features=not args.dont_update_edge_features,
        update_edge_features_pe=not args.dont_update_edge_features_pe,
        rwpe_h=args.rwpe_h,
        rwpe_k=args.rwpe_k,
        cache_rwpe=args.cache_rwpe,
        ortho_embed=args.ortho_embed,
        no_tct=args.no_tct,
        mid_in_edges=args.mid_in_edges,
    )
    agent = Agent.load(path)
    agent.env_specification = env_specification
    agent.agent_specification = agent_specification
    agent.to(agent_specification.device)
    return agent


def load_validator(
    n_j: int,
    n_m: int,
    agent: Agent,
    args: argparse.Namespace,
    validation_envs: list,
) -> AgentValidator:
    """Evaluate the model on random instances of all sizes."""
    affectations, durations = None, None
    eval_problem_description = Description(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        fixed=args.fixed_problem,
        seed=args.seed,
        affectations=affectations,
        durations=durations,
        n_jobs=n_j,
        n_machines=n_m,
        max_duration=args.max_duration,
        duration_mode_bounds=args.duration_mode_bounds,
        duration_delta=args.duration_delta,
    )
    training_specification = TrainingSpecification(
        total_timesteps=args.total_timesteps,
        n_validation_env=args.n_validation_env,
        fixed_validation=args.fixed_validation,
        fixed_random_validation=args.fixed_random_validation,
        validation_batch_size=args.validation_batch_size,
        validation_freq=1 if args.validation_freq == -1 else args.validation_freq,
        display_env="eval-model_wheatley",
        path="/tmp/",
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
    )

    validator = AgentValidator(
        eval_problem_description,
        agent.env_specification,
        args.device,
        training_specification,
        args.disable_visdom,
        validation_envs=validation_envs,
    )
    return validator


def eval_on_instances(agent: Agent, instances: dict, args: argparse.Namespace) -> dict:
    perfs = dict()
    env_specification = agent.env_specification

    for (n_j, n_m), subdir_instances in instances.items():
        if (
            agent.env_specification.max_n_jobs < n_j
            or agent.env_specification.max_n_machines < n_m
        ):
            continue

        print(f"\nBenchmarking {n_j}x{n_m} instances")
        args.n_validation_env = len(subdir_instances)
        args.fixed_validation = True
        problem_description = Description(
            transition_model_config=args.transition_model_config,
            reward_model_config=args.reward_model_config,
            deterministic=(args.duration_type == "deterministic"),
            fixed=args.fixed_problem,
            seed=args.seed,
            affectations=None,
            durations=None,
            n_jobs=n_j,
            n_machines=n_m,
            max_duration=args.max_duration,
            duration_mode_bounds=args.duration_mode_bounds,
            duration_delta=args.duration_delta,
        )

        # Load the validation instances.
        validation_envs = []
        for instance_path in subdir_instances:
            env = Env(problem_description, env_specification)
            env.reset()
            env.state = State.from_instance_file(
                instance_path,
                env_specification.max_n_jobs,
                env_specification.max_n_machines,
                env_specification.n_features,
                deterministic=problem_description.deterministic,
                feature_list=env_specification.input_list,
                observe_conflicts_as_cliques=env_specification.observe_conflicts_as_cliques,
            )
            validation_envs.append(env)

        args.n_validation_env = len(validation_envs)
        validator = load_validator(n_j, n_m, agent, args, validation_envs)
        validator._evaluate_agent(agent)

        perfs[f"{n_j}x{n_m}"] = {
            "PPO": validator.makespans[-1],
            "Random": validator.random_makespans[-1],
        }
        for ortools_strategies, values in validator.ortools_makespans.items():
            perfs[f"{n_j}x{n_m}"][f"OR-Tools - {ortools_strategies}"] = (
                values[-1].cpu().item()
            )
        for custom_agent in validator.custom_agents:
            perfs[f"{n_j}x{n_m}"][custom_agent.rule] = validator.custom_makespans[
                custom_agent.rule
            ][-1]

    return perfs


def list_instances(root_dir: Path) -> Dict[Tuple[int, int], List[Path]]:
    instances = dict()
    for instance_dir in root_dir.iterdir():
        if not instance_dir.is_dir():
            continue

        subdir_instances = []
        for instance_name in instance_dir.iterdir():
            if not str(instance_name).endswith(".npz"):
                continue

            subdir_instances.append(instance_name)

        n_j, n_m = instance_dir.name.split("x")
        instances[(int(n_j), int(n_m))] = subdir_instances

    return instances


def save_perfs(perfs: dict, path: str):
    if not path.endswith("/"):
        path += "/"
    filename = path + "eval.json"

    with open(filename, "w") as f:
        json.dump(perfs, f)

    print(f"Saved perfs to {filename}.")


if __name__ == "__main__":
    from args import argument_parser, parse_args

    parser = argument_parser()
    args, _, _ = parse_args(parser)

    instances = list_instances(Path(f"./instances/jssp/{args.duration_type}"))

    agent = load_agent(args, args.path)
    perfs = eval_on_instances(agent, instances, args)
    save_perfs(perfs, args.path)

    for instance_size, instance_perfs in perfs.items():
        print(f"\n{instance_size} instances:")
        for agent_name, perf in instance_perfs.items():
            print(f"{agent_name}: {perf}")
