"""Evaluate trained model on all instances size."""
import argparse
import json

import torch

from env.jssp_env_specification import JSSPEnvSpecification
from models.agent_specification import AgentSpecification
from models.agent_validator import AgentValidator
from models.jssp_agent import JSSPAgent
from models.training_specification import TrainingSpecification
from problem.jssp_description import JSSPDescription


def load_agent(args: argparse.Namespace, path: str) -> JSSPAgent:
    env_specification = JSSPEnvSpecification(
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
        ortho_embed=args.ortho_embed,
        no_tct=args.no_tct,
    )
    agent = JSSPAgent.load(path)
    agent.env_specification = env_specification
    agent.agent_specification = agent_specification
    return agent


def load_validator(
    n_j: int, n_m: int, agent: JSSPAgent, args: argparse.Namespace
) -> AgentValidator:
    """Evaluate the model on random instances of all sizes."""
    affectations, durations = None, None
    eval_problem_description = JSSPDescription(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        fixed=args.fixed_problem,
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
    )

    validator = AgentValidator(
        eval_problem_description,
        agent.env_specification,
        args.device,
        training_specification,
        args.disable_visdom,
    )
    return validator


def eval_on_instances(
    agent: JSSPAgent, instances: list, args: argparse.Namespace
) -> dict:
    perfs = dict()

    for n_j, n_m in instances:
        print(f"\nBenchmarking {n_j}x{n_m} instances")
        validator = load_validator(n_j, n_m, agent, args)
        validator._evaluate_agent(agent)

        perfs[f"{n_j}x{n_m}"] = {
            "PPO": validator.makespans[-1],
            "OR-tools": validator.ortools_makespans[-1].cpu().item(),
            "Random": validator.random_makespans[-1],
        }
        for custom_agent in validator.custom_agents:
            perfs[f"{n_j}x{n_m}"][custom_agent.rule] = validator.custom_makespans[
                custom_agent.rule
            ][-1]

    return perfs


def save_perfs(perfs: dict, path: str):
    if not path.endswith("/"):
        path += "/"
    filename = path + "eval.json"

    with open(filename, "w") as f:
        json.dump(perfs, f)

    print(f"Saved perfs to {filename}.")


if __name__ == "__main__":
    from args import args

    instances = [
        (6, 6),
        (10, 10),
        (15, 15),
        (20, 20),
    ]

    agent = load_agent(args, args.path)
    perfs = eval_on_instances(agent, instances, args)
    save_perfs(perfs, args.path)

    for instance_size, instance_perfs in perfs.items():
        print(f"\n{instance_size} instances:")
        for agent_name, perf in instance_perfs.items():
            print(f"{agent_name}: {perf}")
