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

import os
import random
import glob

import numpy as np
import torch
import signal

from alg.ppo import PPO
from alg.pretrain import Pretrainer
from generic.utils import get_path
from generic.agent_specification import AgentSpecification
from generic.agent_validator import AgentValidator
from generic.training_specification import TrainingSpecification
from psp.description import Description
from psp.env.env import Env
from psp.env.env_specification import EnvSpecification
from psp.env.genv import GEnv
from psp.models.agent import Agent
from psp.utils.loaders import PSPLoader
from psp.utils.taillard_rcpsp import TaillardRcpsp
from functools import partial

torch.set_float32_matmul_precision("high")


def main(args, exp_name) -> float:
    exp_name = args.exp_name_appendix
    path = get_path(args.path, exp_name)
    torch.distributions.Distribution.set_default_validate_args(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # If we want to load a specific problem, under the taillard (extended) format, and train on it, we first do it.
    # Note that this problem can be stochastic or deterministic
    loader = PSPLoader(generate_bounds=args.generate_duration_bounds)

    assert (
        args.random_taillard
        ^ (args.load_problem is not None)
        ^ (args.train_dir is not None)
    ), "Only provide either --random_taillard, --load_problem or --train_dir to specify the training instances."

    if args.random_taillard:
        psp = TaillardRcpsp(
            pb_id=0,
            n_jssp_jobs=args.n_j,
            n_jssp_machines=args.n_m,
            seed=None,
            generate_bounds=args.generate_duration_bounds,
            duration_mode_bounds=args.duration_mode_bounds,
            duration_delta=args.duration_delta,
            stochastic=args.duration_type == "stochastic",
        )
        train_psps = [psp]

        if args.test_dir is not None:
            test_psps = loader.load_directory(args.test_dir)
        else:
            test_psps = [
                TaillardRcpsp(
                    pb_id=1 + i,
                    n_jssp_jobs=args.n_j,
                    n_jssp_machines=args.n_m,
                    # To make sure the seeds will never be the same.
                    seed=args.seed + args.total_timesteps * (i + 1),
                    generate_bounds=args.generate_duration_bounds,
                    duration_mode_bounds=args.duration_mode_bounds,
                    duration_delta=args.duration_delta,
                    stochastic=(args.duration_type == "stochastic"),
                )
                for i in range(args.n_validation_env)
            ]
    elif args.load_problem is None:
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
    # train_psps = [TaillardRcpsp(id, 6, 6, id) for id in range(10)]
    # test_psps = [TaillardRcpsp(id + 10, 6, 6, id + 10) for id in range(10)]

    # Define problem and visualize it
    problem_description = Description(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.criterion,
        deterministic=(args.duration_type == "deterministic"),
        train_psps=train_psps,
        test_psps=test_psps,
        seed=args.seed,
    )
    problem_description.print_self()

    # Then specify the variables used for the training

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
        store_rollouts_on_disk=args.store_rollouts_on_disk,
        critic_loss=args.critic_loss,
        debug_net=args.debug_net,
        display_gantt=args.display_gantt,
        max_shared_mem_per_worker=args.max_shared_mem_per_worker,
        espo=args.espo,
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
        # pyg=args.pyg or args.hierarchical or args.tokengt,
        pyg=True,
        max_n_modes=args.max_n_modes,
    )
    env_specification.print_self()
    if args.batch_size == 1 and not args.dont_normalize_advantage:
        print(
            "batch size 1 and normalize advantage are not compatible\neither set --batch_size to > 1  or append --dont_normalize_advantage"
        )
        exit()

    if args.two_hot is not None:
        hidden_dim_critic = max(int(args.two_hot[2]), args.hidden_dim_critic)
    elif args.hl_gauss is not None:
        hidden_dim_critic = max(int(args.hl_gauss[2]), args.hidden_dim_critic)
    else:
        hidden_dim_critic = args.hidden_dim_critic
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
        hidden_dim_critic=hidden_dim_critic,
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
        # pyg=args.pyg or args.hierarchical or args.tokengt,
        pyg=True,
        hierarchical=args.hierarchical,
        tokengt=args.tokengt,
        shared_conv=args.shared_conv,
        checkpoint=args.checkpoint,
        dual_net=args.dual_net,
    )
    agent_specification.print_self()

    # If we want to use a pretrained Agent, we only have to load it (if it exists)
    if args.retrain and os.path.exists(path + "/agent.pkl"):
        print("Retraining an already existing agent\n")
        agent = Agent.load(path)
        agent.env_specification = env_specification
        agent.agent_specification = agent_specification
    elif (
        args.resume
        and os.path.exists(path + "/agent.pkl")
        and os.path.exists(path + "/optimizer.pkl")
    ):
        print("Resuming a training\n")
        agent = Agent.load(path, graphobs=args.vecenv_type == "graphgym")
        opt_state_dict = torch.load(path + "/optimizer.pkl")
        agent.env_specification = env_specification
        agent.agent_specification = agent_specification
    else:
        agent = Agent(
            env_specification=env_specification,
            agent_specification=agent_specification,
            graphobs=args.vecenv_type == "graphgym",
        )

    if args.pretrain:
        agent.to(args.device)
        pretrainer = Pretrainer(
            problem_description,
            env_specification,
            training_specification,
            Env,
            num_envs=args.pretrain_num_envs,
            prob=args.pretrain_prob,
        )
        pretrainer.pretrain(
            agent,
            args.pretrain_epochs,
            args.pretrain_batch_size,
            args.pretrain_n_steps_episode,
            lr=args.pretrain_lr,
        )

    # And finally, we train the model on the specified training mode
    # Note: The saving of the best model is hanlded in the agent.train method.
    # We save every time we hit a min RL / OR-Tools ratio
    # agent.train(problem_description, training_specification)

    validator = AgentValidator(
        problem_description,
        env_specification,
        args.device,
        training_specification,
        args.disable_visdom,
        graphobs=args.vecenv_type == "graphgym",
        compute_ortools=not args.disable_ortools,
    )
    if args.vecenv_type == "graphgym":
        env_cls = GEnv
    else:
        env_cls = Env
    ppo = PPO(training_specification, env_cls, validator, args.generate_duration_bounds)
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
        warmup=args.warmup,
    )


def interrupt_handler(path, signum, frame):
    if path is not None:
        files = glob.glob(path + "/wheatley_" + str(os.getpid()) + "_*.obs")
        print("cleaning observations ")
        for f in files:
            os.remove(f)
            print(".", end="")
    print()
    exit()


if __name__ == "__main__":
    from args import argument_parser, parse_args

    print("installing cleanup handler")
    parser = argument_parser()
    args, exp_name = parse_args(parser)

    signal.signal(
        signal.SIGINT, partial(interrupt_handler, args.store_rollouts_on_disk)
    )

    main(args, exp_name)
