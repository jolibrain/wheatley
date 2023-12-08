#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Pierre Peirera <pierre.peirera@jolibrain.com>
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

import os

import numpy as np
import torch

from alg.ppo import PPO
from alg.pretrain import Pretrainer
from generic.agent_specification import AgentSpecification
from generic.agent_validator import AgentValidator
from generic.training_specification import TrainingSpecification
from jssp.description import Description as ProblemDescription
from jssp.env.env import Env
from jssp.env.env_specification import EnvSpecification
from jssp.models.agent import Agent
from jssp.utils.utils import load_problem


def main(args, exp_name, path) -> float:
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # If we want to load a specific problem, under the taillard (extended) format, and train on it, we first do it.
    # Note that this problem can be stochastic or deterministic
    affectations, durations = None, None
    if args.load_problem is not None:
        args.n_j, args.n_m, affectations, durations = load_problem(
            args.load_problem,
            taillard_offset=args.first_machine_id_is_one,
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
        seed=args.seed,
    )
    problem_description.print_self()

    # Specific evaluation problem, can be different from the training problem
    if (
        args.fixed_problem
        and args.fixed_validation
        and args.duration_type == "deterministic"
        and args.eval_n_j == args.n_j
        and args.eval_n_m == args.n_m
    ):
        eval_problem_description = problem_description
    else:
        eval_problem_description = ProblemDescription(
            transition_model_config=args.transition_model_config,
            reward_model_config=args.reward_model_config,
            deterministic=(args.duration_type == "deterministic"),
            fixed=args.fixed_problem,
            affectations=affectations,
            durations=durations,
            n_jobs=args.eval_n_j,
            n_machines=args.eval_n_m,
            max_duration=args.max_duration,
            duration_mode_bounds=args.duration_mode_bounds,
            duration_delta=args.duration_delta,
            seed=args.seed,
        )

    # Then specify the variables used for the training

    training_specification = TrainingSpecification(
        total_timesteps=args.total_timesteps,
        n_validation_env=args.n_validation_env,
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
    )
    training_specification.print_self()

    opt_state_dict = None

    if args.conflicts == "clique" and args.precompute_cliques:
        observe_clique = True
    else:
        observe_clique = False
    if args.observe_duration_when_affect:
        observe_real_duration_when_affect = True
    else:
        observe_real_duration_when_affect = False
    env_specification = EnvSpecification(
        max_n_jobs=args.max_n_j,
        max_n_machines=args.max_n_m,
        normalize_input=not args.dont_normalize_input,
        input_list=args.features,
        insertion_mode=args.insertion_mode,
        max_edges_factor=args.max_edges_upper_bound_factor,
        sample_n_jobs=args.sample_n_jobs,
        chunk_n_jobs=args.chunk_n_jobs,
        observe_conflicts_as_cliques=observe_clique,
        observe_real_duration_when_affect=observe_real_duration_when_affect,
        do_not_observe_updated_bounds=args.do_not_observe_updated_bounds,
    )
    env_specification.print_self()
    if args.batch_size == 1 and not args.dont_normalize_advantage:
        print(
            "batch size 1 and normalize advantage are not compatible\neither set --batch_size to > 1  or append --dont_normalize_advantage"
        )
        exit()
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
        ortho_embed=args.ortho_embed,
        no_tct=args.no_tct,
        mid_in_edges=args.mid_in_edges,
        rwpe_k=args.rwpe_k,
        rwpe_h=args.rwpe_h,
        cache_rwpe=args.cache_rwpe,
        two_hot=args.two_hot,
        symlog=args.symlog,
    )
    agent_specification.print_self()
    # If we want to use a pretrained Agent, we only have to load it (if it exists)
    if args.retrain:
        print("Retraining an already existing agent\n")
        agent = Agent.load(args.retrain)
        agent.env_specification = env_specification
        agent.agent_specification = agent_specification
    elif (
        args.resume
        and os.path.exists(path + "/agent.pkl")
        and os.path.exists(path + "/optimizer.pkl")
    ):
        print("Resuming a training\n")
        agent = Agent.load(path)
        opt_state_dict = torch.load(path + "/optimizer.pkl")
        agent.env_specification = env_specification
        agent.agent_specification = agent_specification
    else:
        agent = Agent(
            env_specification=env_specification, agent_specification=agent_specification
        )

    if args.pretrain:
        agent.to(args.device)
        pretrainer = Pretrainer(
            problem_description,
            env_specification,
            training_specification,
            Env,
            num_envs=args.pretrain_num_envs,
            num_eval_envs=args.pretrain_num_eval_envs,
            trajectories=args.pretrain_trajectories,
            prob=args.pretrain_prob,
        )
        pretrainer.pretrain(
            agent,
            args.pretrain_epochs,
            args.pretrain_batch_size,
            lr=args.pretrain_lr,
            vf_coeff=args.pretrain_vf_coef,
            weight_decay=args.pretrain_weight_decay,
        )

    if args.reinit_head_before_ppo:
        agent.init_heads()

    # And finally, we train the model on the specified training mode
    # Note: The saving of the best model is handled in the agent.train method.
    # We save every time we hit a min RL / OR-Tools ratio
    # agent.train(problem_description, training_specification)

    validator = AgentValidator(
        eval_problem_description,
        env_specification,
        args.device,
        training_specification,
        args.disable_visdom,
    )
    if args.resume and os.path.exists(path + "/validator.pkl"):
        validator = validator.reload_state(path + "/validator.pkl")
        print("Validator reloaded.")

    ppo = PPO(
        training_specification,
        Env,
        validator,
    )
    return ppo.train(
        agent,
        problem_description,
        env_specification,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=1,
        train_device=args.device,
        rollout_agent_device=args.device,
        opt_state_dict=None,
        skip_initial_eval=args.skip_initial_eval,
        skip_model_trace=args.skip_model_trace,
    )


if __name__ == "__main__":
    from args import argument_parser, parse_args

    parser = argument_parser()
    args, exp_name, path = parse_args(parser)
    main(args, exp_name, path)
