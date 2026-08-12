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

import torch
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
from pp.description import Description
from pp.domains.registry import get_domain_definition
from pp.models.agent import Agent
from pp.utils.utils import create_train_envs
import torchinfo
from functools import partial
from pp.graph.pyg_graph import PYGGraph


torch.set_float32_matmul_precision("high")


def main(args) -> float:
    exp_name = args.exp_name_appendix
    domain_def = get_domain_definition(args.domain)
    path = get_path(args.path, exp_name)
    torch.distributions.Distribution.set_default_validate_args(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    generator_kwargs = {}
    if domain_def.name == "maze":
        generator_kwargs["maze_gen"] = args.pp_maze_gen
    elif domain_def.name == "navigation":
        generator_kwargs["num_goals"] = args.pp_num_goals
        generator_kwargs["random_num_goals"] = args.pp_random_num_goals
        generator_kwargs["seed"] = args.seed

    train_pp, test_pp = domain_def.generator_fn(
        args.maze_size_train if args.pp_maze_gen != "maze_hard" else 30,
        args.n_train_mazes,
        args.maze_size_test if args.pp_maze_gen != "maze_hard" else 30,
        args.n_test_mazes,
        **generator_kwargs,
    )
    problem_description = Description(
        args.transition_model_config,
        args.reward_model_config,
        train_pp,
        test_pp,
        args.seed,
    )
    problem_description.print_self()

    training_specification = TrainingSpecification(
        total_timesteps=args.total_timesteps,
        n_validation_env=len(problem_description.test_pbs),
        fixed_validation=args.fixed_validation,
        fixed_random_validation=args.fixed_random_validation,
        no_random_validation=args.no_random_validation,
        validation_batch_size=args.validation_batch_size,
        validation_freq=1 if args.validation_freq == -1 else args.validation_freq,
        display_env=exp_name,
        path=path,
        custom_heuristic_names=args.custom_heuristic_names,
        ortools_strategy=args.ortools_strategy,
        max_time_ortools=args.max_time_ortools,
        scaling_constant_ortools=args.scaling_constant_ortools,
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
        clip_range_high=args.clip_range_high,
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
        clip_grad_norm=args.clip_grad_norm,
        anon_gamma=args.anon_gamma,
        spo=args.spo,
    )
    training_specification.print_self()

    env_specification = domain_def.env_spec_cls(
        normalize_input=True, max_n_nodes=args.maze_size_train * args.maze_size_train
    )
    env_specification.print_self()

    agent_specification = AgentSpecification(
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
        residual_gnn=args.residual_gnn if args.hierarchical else True,
        # normalize_gnn=True if args.hierarchical else args.normalize_gnn,
        normalize_gnn=args.normalize_gnn,
        n_mlp_layers_actor=args.n_mlp_layers_actor,
        hidden_dim_actor=args.hidden_dim_actor,
        n_mlp_layers_critic=args.n_mlp_layers_critic,
        hidden_dim_critic=args.hidden_dim_critic,
        dropout=args.dropout,
        cache_lap_node_id=not args.dont_cache_lap_node_id,
        lap_node_id_k=args.lap_node_id_k,
        vnode=args.vnode,
        update_edge_features=args.update_edge_features,
        update_edge_features_pe=args.update_edge_features_pe,
        ortho_embed=args.ortho_embed,
        no_tct=args.no_tct,
        rwpe_k=args.rwpe_k,
        rwpe_h=args.rwpe_h,
        cache_rwpe=args.cache_rwpe,
        two_hot=args.two_hot,
        symlog=args.symlog,
        hl_gauss=args.hl_gauss,
        reward_weights=args.reward_weights,
        sgformer=args.sgformer,
        # pyg=args.pyg or args.hierarchical or args.tokengt,
        hierarchical=args.hierarchical,
        shared_conv=args.shared_conv,
        checkpoint=args.checkpoint,
        dual_net=args.dual_net,
        lappe=args.lappe,
        rwpe=args.rwpe,
        bidir=args.bidir,
        self_loops=args.self_loops if args.hierarchical else True,
        gconv_activation=args.gconv_activation,
        g2=args.g2,
        nonchrono=args.nonchrono,
        agent_types=args.pp_agent_types,
    )
    agent_specification.print_self()

    opt_state_dict = None
    if (
        args.resume is not None
        and os.path.exists(args.resume + "/agent.pkl")
        and os.path.exists(args.resume + "/optimizer.pkl")
    ):
        print("Resuming a training\n")
        agent = Agent.load(
            args.resume + "/", max_n_modes=args.maze_size_train * args.maze_size_train
        )
        agent.gnn.to(torch.device(args.device))
        agent.value_net.to(torch.device(args.device))
        for head in agent.action_nets:
            head.to(torch.device(args.device))
        opt_state_dict = torch.load(args.resume + "/optimizer.pkl")
        agent.env_specification = env_specification
        agent.agent_specification = agent_specification
    else:
        if not args.skip_model_trace:
            agent = Agent(
                env_specification=env_specification,
                agent_specification=agent_specification,
                do_compile=False,
            )
            torchinfo.summary(agent, depth=3, verbose=1)
        agent = Agent(
            env_specification=env_specification,
            agent_specification=agent_specification,
        )

    if args.reinit_head_before_ppo:
        agent.init_heads()

    pp_env_kwargs = {
        "pp_agent_types": args.pp_agent_types,
        # "pp_allow_stay": args.pp_allow_stay,
        # "pp_all_agents_must_finish": args.pp_all_agents_must_finish,
        "pp_maze_gen": args.pp_maze_gen,
        # "pp_force_common_start": args.pp_force_common_start,
        # "pp_num_goals": args.pp_num_goals,
        # "pp_random_num_goals": args.pp_random_num_goals,
        # "pp_danger_max_size": args.pp_danger_max_size,
        # "pp_danger_max_num": args.pp_danger_max_num,
        # "pp_danger_prob": args.pp_danger_prob,
        # "pp_danger_multiplier": args.pp_danger_multiplier,
        "pp_goal_reward": args.pp_goal_reward,
        # "pp_k_lookahead": args.pp_k_lookahead,
        # "pp_heading_motion": args.pp_heading_motion,
        # "pp_max_turn": args.pp_max_turn,
        # "pp_protect_max_num": args.pp_protect_max_num,
        # "pp_protect_max_radius": args.pp_protect_max_radius,
        # "pp_protect_kill_prob": args.pp_protect_kill_prob,
        "nonchrono": args.nonchrono,
        "walls": not args.no_walls,
        "domain_name": domain_def.name,
    }

    validator = AgentValidator(
        problem_description,
        env_specification,
        args.device,
        env_cls=domain_def.env_cls,
        training_specification=training_specification,
        disable_visdom=args.disable_visdom,
        lappe=agent_specification.lappe,
        rwpe=agent_specification.rwpe,
        env_kwargs=pp_env_kwargs,
    )
    # if args.resume is not None and os.path.exists(path + "validator.pkl"):
    #     validator = validator.reload_state(path + "validator.pkl")
    #     print("Validator reloaded.")
    ppo = PPO(training_specification, validator, obstype=PYGGraph)

    train_envs = create_train_envs(
        env_cls=domain_def.env_cls,
        problem_description=problem_description,
        env_specification=env_specification,
        num_envs=training_specification.n_workers,
        max_shared_mem_per_worker=training_specification.max_shared_mem_per_worker,
        lappe=agent_specification.lappe,
        rwpe=agent_specification.rwpe,
        create_new_pbs=args.infinite_dataset,
        env_kwargs=pp_env_kwargs,
    )

    return ppo.train(
        agent,
        problem_description,
        env_specification,
        train_envs,
        training_specification.n_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=1,
        train_device=args.device,
        rollout_agent_device=args.device,
        opt_state_dict=opt_state_dict,
        skip_initial_eval=args.skip_initial_eval,
        warmup=args.warmup,
        laber=args.laber,
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
    from pp.args_pp import argument_parser, parse_args

    parser = argument_parser()
    args = parse_args(parser)

    if args.store_rollouts_on_disk is not None:
        print("Installing cleanup handler for rollouts on disk")
        signal.signal(
            signal.SIGINT, partial(interrupt_handler, args.store_rollouts_on_disk)
        )

    main(args)
