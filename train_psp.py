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

import numpy as np
import torch

from alg.ppo import PPO
from alg.pretrain import Pretrainer
from args import args, exp_name, get_path, path
from env.psp_env import PSPEnv
from env.psp_env_specification import PSPEnvSpecification
from models.agent_specification import AgentSpecification
from models.agent_validator import AgentValidator
from models.psp_agent import PSPAgent as Agent
from models.training_specification import TrainingSpecification
from problem.psp_description import PSPDescription
from utils.loaders import PSPLoader
from utils.utils import generate_deterministic_problem, generate_problem_distrib


def main():
    exp_name = args.exp_name_appendix
    path = get_path(args.path, exp_name)
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # If we want to load a specific problem, under the taillard (extended) format, and train on it, we first do it.
    # Note that this problem can be stochastic or deterministic
    loader = PSPLoader(generate_bounds=args.generate_duration_bounds)
    psp = loader.load_single(args.load_problem)
    train_psps = None
    test_psps = None

    # Define problem and visualize it
    problem_description = PSPDescription(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        train_psps=[psp],
        test_psps=[psp],
    )
    problem_description.print_self()

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
        custom_heuristic_name=args.custom_heuristic_name,
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
    )
    training_specification.print_self()

    opt_state_dict = None
    # If we want to use a pretrained Agent, we only have to load it (if it exists)
    if args.retrain and os.path.exists(path + ".agent"):
        print("Retraining an already existing agent\n")
        agent = Agent.load(path)
    if (
        args.resume
        and os.path.exists(path + ".agent")
        and os.path.exists(path + ".opt")
    ):
        print("Resuming a training\n")
        agent = Agent.load(path)
        opt_state_dict = torch.load(path + ".opt")

    # Else, we instantiate a new Agent
    else:
        if args.conflicts == "clique" and args.precompute_cliques:
            observe_clique = True
        else:
            observe_clique = False
        if args.observe_duration_when_affect:
            observe_real_duration_when_affect = True
        else:
            observe_real_duration_when_affect = False
        env_specification = PSPEnvSpecification(
            problems=problem_description,
            normalize_input=not args.dont_normalize_input,
            input_list=args.features,
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
        )
        agent_specification.print_self()
        agent = Agent(
            env_specification=env_specification, agent_specification=agent_specification
        )

    if args.pretrain:
        agent.to(args.device)
        pretrainer = Pretrainer(
            problem_description,
            env_specification,
            training_specification,
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
    )
    ppo = PPO(
        training_specification,
        PSPEnv,
        validator,
    )
    ppo.train(
        agent,
        problem_description,
        env_specification,
        lr=args.lr,
        log_interval=1,
        train_device=args.device,
        rollout_agent_device=args.device,
        opt_state_dict=None,
        skip_initial_eval=args.skip_initial_eval,
    )


if __name__ == "__main__":
    exp_name = "JSSP"
    main()
