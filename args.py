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

import argparse
from typing import Tuple

from generic.utils import get_exp_name
from jssp.dispatching_rules.heuristics import HEURISTICS


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="These args can be used with train.py, test.py and benchmark/run_taillard.py. They specify how the training"
        "(or testing) is going to be performed"
    )

    # =================================================PROBLEM DESCRIPTION======================================================
    parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
    parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
    parser.add_argument(
        "--eval_n_j", type=int, required=False, help="Number of jobs for eval"
    )
    parser.add_argument(
        "--eval_n_m", type=int, required=False, help="Number of machines for eval"
    )
    parser.add_argument(
        "--max_duration", type=int, default=99, help="Max duration for problems"
    )

    # =================================================COMPUTER SPECIFICATION===================================================
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_n_j", type=int, default=-1, help="Max number of jobs (if -1, max_n_j=n_j"
    )
    parser.add_argument(
        "--max_n_m",
        type=int,
        default=-1,
        help="Max number of machines (if -1, max_n_m=n_m",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="saved_networks/",
        help="Path to saved network (default is set to exp_name)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=10,
        help="Number of CPU cores for simulating environment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="Which device to use (cpu, cuda:0, cuda:1...)",
    )
    parser.add_argument(
        "--store_rollouts_on_disk",
        default=None,
        type=str,
        help="location for rollout on disk store rollouts on disk (graphgym only ATM)",
    )
    parser.add_argument(
        "--exp_name_appendix", type=str, help="Appendix for the name of the experience"
    )
    parser.add_argument(
        "--vecenv_type",
        type=str,
        default="graphgym",
        choices=["subproc", "dummy", "graphgym"],
        help="everything deprecated but graphgym",
    )

    # =================================================TRAINING SPECIFICATION====================================================

    parser.add_argument(
        "--espo", default=False, action="store_true", help="use espo instead of PPO"
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=int(1e6),
        help="Number of training env timesteps",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs for updating the agent's parameters",
    )
    parser.add_argument(
        "--n_steps_episode", type=int, default=1024, help="Number of steps per episode."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size during training of the agent",
    )
    parser.add_argument(
        "--iter_size",
        type=int,
        default=1,
        help="iter size during training of the agent",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Default Learning rate")
    parser.add_argument(
        "--fe_lr", type=float, default=None, help="Learning rate for feature extractor"
    )
    parser.add_argument(
        "--rpo", default=False, action="store_true", help="use RPO-style smoothing"
    )
    parser.add_argument(
        "--rpo_smoothing_param",
        type=float,
        default=1.0,
        help="RPO-style smoothing param",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=1.0,
        help="GAE lambda parameter, GAE off by default",
    )
    parser.add_argument(
        "--return_based_scaling",
        default=False,
        action="store_true",
        help="use return based scaling 2105.05347",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="radam",
        choices=["adam", "sgd", "adamw", "radam", "dadam", "lion"],
        help="Which optimizer to use",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="PPO weight decay",
    )
    parser.add_argument(
        "--freeze_graph",
        default=False,
        action="store_true",
        help="Freezes graph during training",
    )
    parser.add_argument(
        "--custom_heuristic_names",
        choices=list(HEURISTICS.keys()),
        nargs="*",
        help="Which custom heuristic to run",
    )
    parser.add_argument(
        "--retrain",
        type=str,
        default="",
        help="Use this flag if you want to retrain a trained model. You must provide the direct path to the model you want to load.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help='Resume a previous training. The script will look for trained model named "agent.pkl" in the directory experiment.',
    )
    parser.add_argument(
        "--reinit_head_before_ppo",
        default=False,
        action="store_true",
        help="Remove existing head (from a pretrain, resume or retrain) and initialize a new head before starting PPO",
    )
    parser.add_argument(
        "--debug_net",
        default=False,
        action="store_true",
        help="collect and display statistics about net",
    )
    parser.add_argument(
        "--checkpoint",
        default=1,
        type=int,
        help="keep 1 out of checkpoint (this args)  layers in memory during forward pass",
    )

    parser.add_argument("--warmup", default=0, type=int, help="warmup steps")

    # =================================================VALIDATION SPECIFICATION=================================================
    parser.add_argument(
        "--n_validation_env",
        type=int,
        default=20,
        help="Number of validation environments ",
    )
    parser.add_argument(
        "--fixed_validation",
        action="store_true",
        help="Use the same problems/durations sampling and OR-Tools solutions",
    )
    parser.add_argument(
        "--fixed_random_validation",
        type=int,
        default=0,
        help="Average the random solutions over N random runs, requires --fixed_validation",
    )
    parser.add_argument(
        "--validation_freq",
        type=int,
        default=-1,
        help="Number of steps between each evaluation",
    )
    parser.add_argument(
        "--max_time_ortools",
        type=int,
        default=3,
        help="Max compute time for ortools (in seconds)",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size for predictions of actions during validation",
    )

    # =================================================TESTING SPECIFICATION====================================================
    parser.add_argument(
        "--n_test_problems",
        type=int,
        default=100,
        help="Number of problems for testing",
    )
    parser.add_argument(
        "--test_print_every", type=int, default=50, help="Print frequency for testing"
    )

    # =================================================AGENT SPECIFICATION======================================================
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument(
        "--clip_range", type=float, default=0.25, help="Clipping parameter"
    )
    # parser.add_argument("--clip_range", type=float, default=None, help="Clipping parameter")

    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.04,
        help="Limit the KL divergence between updates",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.0, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient"
    )

    parser.add_argument(
        "--critic_loss",
        type=str,
        choices=["l2", "l1", "l1w", "l1ws"],
        default="l2",
        help="critic loss",
    )
    parser.add_argument(
        "--reward_weights",
        type=float,
        nargs="+",
        default=[1],
        help="reward weights (default: [1] : unidimensional reward)",
    )
    parser.add_argument(
        "--dont_normalize_advantage",
        action="store_true",
        help="Whether to not normalize PPO advantage",
    )

    parser.add_argument(
        "--gconv_type",
        type=str,
        default="gatv2",
        choices=["gin", "gatv2", "pna", "dgn", "gcn2", "pdf"],
        help="Graph convolutional neural network type: gin for GIN, gatv2 for GATV2",
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="learn",
        choices=["max", "avg", "learn", "learninv", "gap"],
        help="which pooling to use (avg , max or learn or gap)",
    )
    parser.add_argument(
        "--layer_pooling",
        type=str,
        default="all",
        choices=["last", "all"],
        help="use all or only last layer as node value, used only in tokengt",
    )
    parser.add_argument(
        "--mlp_act_graph",
        type=str,
        default="gelu",
        choices=["relu", "tanh", "elu", "gelu", "selu", "silu"],
        help="agent mlp extractor activation type",
    )
    parser.add_argument(
        "--mlp_act",
        type=str,
        default="gelu",
        choices=["relu", "tanh", "elu", "gelu", "selu"],
        help="agent mlp extractor activation type",
    )
    parser.add_argument(
        "--sgformer", default=False, action="store_true", help="add sgformer to network"
    )
    # parser.add_argument(
    #     "--pyg", default=True, action="store_true", help="use pyg instead of DGL"
    # )
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout ratio")
    parser.add_argument(
        "--ortools_strategy",
        type=str,
        nargs="*",
        choices=["realistic", "optimistic", "pessimistic", "averagistic"],
        help="ortools durations estimations in pessimistic|optimistic|averagistic|realistic realistic means omiscient, "
        "ie sees the future",
    )
    parser.add_argument(
        "--fe_type",
        type=str,
        default="message_passing",
        help="feature extractor type in [message_passing|tokengt]",
        choices=["message_passing", "tokengt"],
    )
    parser.add_argument(
        "--transformer_flavor",
        type=str,
        default="linear",
        help="transformer implementation for tokengt",
        choices=["vanilla", "linear", "performer"],
    )
    parser.add_argument(
        "--performer_nb_features",
        type=int,
        default=None,
        help="number of projections features for performer (for tokengt), default is n.log(n), where n is head dim",
    )
    parser.add_argument(
        "--performer_redraw_interval",
        type=int,
        default=1000,
        help="redraw interval for features basis  for performer (for tokengt)",
    )
    parser.add_argument(
        "--performer_generalized_attention",
        action="store_true",
        default=False,
        help="generalized attention  for performer (for tokengt)",
    )
    parser.add_argument(
        "--performer_auto_check_redraw",
        default=False,
        action="store_true",
        help="auto check redraw for performer (for tokengt)",
    )

    parser.add_argument(
        "--dont_cache_lap_node_id",
        action="store_true",
        help="disable laplacian cache for tokengt",
    )
    parser.add_argument(
        "--lap_node_id_k", type=int, default=10, help="laplacian id size for tokengt"
    )
    parser.add_argument(
        "--graph_has_relu",
        action="store_true",
        help="whether graph feature extractor has activations between layers",
    )
    parser.add_argument(
        "--n_mlp_layers_features_extractor",
        type=int,
        default=3,
        help="Number of MLP layers in each GNN",
    )
    parser.add_argument(
        "--n_layers_features_extractor",
        type=int,
        default=6,
        help="Number of layers of GNN",
    )
    parser.add_argument(
        "--hidden_dim_features_extractor",
        type=int,
        default=64,
        help="Dimension of hidden and output for GNN",
    )
    parser.add_argument(
        "--n_attention_heads",
        type=int,
        default=4,
        help="Number of heads for internal attention",
    )
    parser.add_argument(
        "--reverse_adj_in_gnn", action="store_true", help="reverse adj matrix in GNN"
    )
    parser.add_argument(
        "--residual_gnn", action="store_true", help="use residual connection in GNN"
    )
    parser.add_argument(
        "--normalize_gnn", action="store_true", help="normalize gnn everywhere"
    )
    parser.add_argument(
        "--conflicts",
        type=str,
        help="machine conflict encoding in [att|clique|node]",
        default="clique",
        choices=["att", "clique", "node"],
    )
    parser.add_argument(
        "--no_tct",
        default=False,
        action="store_true",
        help="do not explicitly compute/use tct before gnn",
    )

    parser.add_argument(
        "--mid_in_edges",
        default=False,
        action="store_true",
        help="add machine id in edge type",
    )
    parser.add_argument(
        "--add_rp_edges",
        default="frontier_strict",
        choices=["all", "frontier", "frontier_strict", "none"],
        help="take into account resource precedence edges",
    )

    parser.add_argument(
        "--remove_old_resource_info",
        default=False,
        action="store_true",
        help="do not take into account already affected task resource info",
    )

    parser.add_argument(
        "--keep_past_prec",
        default=False,
        action="store_true",
        help="keep past precedencies",
    )
    parser.add_argument(
        "--observation_horizon_step",
        default=0,
        type=int,
        help="observation horizon (steps)",
    )
    parser.add_argument(
        "--observation_horizon_time",
        default=0,
        type=float,
        help="observation horizon (time)",
    )
    parser.add_argument(
        "--fast_forward",
        default=False,
        action="store_true",
        help="make env auto forward trivial actions",
    )

    parser.add_argument(
        "--observe_subgraph",
        default=False,
        action="store_true",
        help="extract subgraph (graphgym only ATM)",
    )
    parser.add_argument(
        "--random_taillard",
        action="store_true",
        help="generate and use random jssp taillard instances for rcpsp training",
    )

    parser.add_argument(
        "--vnode", default=False, action="store_true", help="add vnode to MP-graph"
    )

    parser.add_argument(
        "--update_edge_features",
        default=False,
        action="store_true",
        help="update edge features",
    )
    parser.add_argument(
        "--update_edge_features_pe",
        default=False,
        action="store_true",
        help="update edge features of pe part",
    )
    parser.add_argument(
        "--ortho_embed",
        default=False,
        action="store_true",
        help="init nn.Embeddings with ortho init",
    )

    parser.add_argument(
        "--precompute_cliques",
        default=False,
        action="store_true",
        help="precompute cliques, trades mem with cpu time",
    )
    parser.add_argument(
        "--n_mlp_layers_actor",
        type=int,
        default=1,
        help="Number of MLP layers in actor (excluding input and output",
    )
    parser.add_argument(
        "--hidden_dim_actor", type=int, default=64, help="Hidden dim for actor"
    )
    parser.add_argument(
        "--n_mlp_layers_critic",
        type=int,
        default=1,
        help="Number of MLP layers in critic (excluding input and output)",
    )
    parser.add_argument(
        "--hidden_dim_critic", type=int, default=64, help="Hidden dim for critic"
    )
    parser.add_argument(
        "--edge_embedding_flavor",
        type=str,
        default="sum",
        choices=["sum", "cat"],
        help="edge embedding technique for RCPSP",
    )

    parser.add_argument(
        "--rwpe_k",
        type=int,
        default=0,
        help="number of hops for rwpe (0 for no rwpe)",
    )
    parser.add_argument(
        "--rwpe_h",
        type=int,
        default=16,
        help="hidden dim of pe (times number of subgraphs)",
    )

    parser.add_argument(
        "--cache_rwpe",
        default=False,
        action="store_true",
        help="enable rwpe cache",
    )

    parser.add_argument(
        "--two_hot",
        default=None,
        type=float,
        nargs=3,
        help="min,max, nbins parameters for value two hot encoding",
    )
    parser.add_argument(
        "--hl_gauss",
        default=None,
        type=float,
        nargs=3,
        help="min,max, nbins parameters for value hl_gauss encoding",
    )
    parser.add_argument(
        "--symlog",
        action="store_true",
        default=False,
        help="predict value internally as log of expected sum of reward",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=False,
        help="use hierarchical GNN",
    )
    parser.add_argument(
        "--tokengt",
        action="store_true",
        default=False,
        help="use tokenGT",
    )
    parser.add_argument(
        "--shared_conv",
        action="store_true",
        default=False,
        help="use same conv params across levels",
    )
    parser.add_argument(
        "--dual_net",
        action="store_true",
        default=False,
        help="use two gnn",
    )

    # =================================================ENVIRONMENT SPECIFICATION================================================
    parser.add_argument(
        "--duration_type",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="Specify if the JSSP should be deterministic or stochastic. If stochastic, we generate many problem from same "
        + "distribution data (given in duration_delta and duration_mode_bounds)",
    )
    parser.add_argument(
        "--transition_model_config",
        type=str,
        default="simple",
        choices=["simple", "L2D", "SlotLocking"],
        help="Which transition model to use",
    )
    parser.add_argument(
        "--observe_duration_when_affect",
        default=False,
        action="store_true",
        help="observe real duration at affectatio type, for more efficient replanning",
    )
    parser.add_argument(
        "--do_not_observe_updated_bounds",
        default=False,
        action="store_true",
        help="do not observe task completion time",
    )

    parser.add_argument(
        "--factored_rp",
        default=False,
        action="store_true",
        help="factor resource priority link (automatically used  for tokengt)",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default="makespan",
        choices=["makespan", "tardiness"],
        help="psp criterion in makespan|tardiness",
    )

    parser.add_argument(
        "--reward_model_config",
        type=str,
        default="Sparse",
        choices=[
            "L2D",
            "L2D_optimistic",
            "L2D_pessimistic",
            "L2D_averagistic",
            "Sparse",
            "Tassel",
            "Intrinsic",
            "realistic",
            "optimistic",
            "pessimistic",
            "averagistic",
        ],
        help="Which reward model to use, from L2D|Sparse|Tassel|Intrinsic in the deterministic case; "
        "for uncertainty (stochastic), you can use pessimistic|optimistic|realistic|averagistic|Sparse",
    )
    parser.add_argument(
        "--duration_mode_bounds",
        type=int,
        nargs=2,
        default=(10, 50),
        help="The define the range of sampling for the mode of the triangular distributions for durations",
    )
    parser.add_argument(
        "--duration_delta",
        type=int,
        nargs=2,
        default=(10, 200),
        help="This defines the delta between low_value/high_value and the mode for the triangular distributions for durations",
    )
    parser.add_argument(
        "--insertion_mode",
        type=str,
        default="no_forced_insertion",
        choices=[
            "no_forced_insertion",
            "full_forced_insertion",
            "choose_forced_insertion",
            "slot_locking",
        ],
        help="This defines how the jobs are inserted in the schedule.",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=[
            "duration",
            # "selectable", is mandatory , no need to put it here
            # "one_hot_machineid", is also mandatory, no need to put it here
            # "total_job_time",
            # "total_machine_time",
            # "job_completion_percentage",
            # "machine_completion_percentage",
            # "mopnr",
            # "mwkr",
        ],
        help="The features we want to have as input of features_extractor. Should be in {duration, one_hot_job_id, "
        + "one_hot_machine_id, total_job_time, total_machine_time, job_completion_percentage, machine_completion_percentage, "
        + "mopnr, mwkr",
    )
    parser.add_argument(
        "--dont_normalize_input",
        default=False,
        action="store_true",
        help="Default is dividing input by constant",
    )
    parser.add_argument(
        "--fixed_problem",
        default=False,
        action="store_true",
        help="Fix affectations and durations",
    )
    parser.add_argument(
        "--max_edges_upper_bound_factor",
        type=int,
        default=4,
        help="Upper bound factor to max_n_edges, allows lowering the overall memory usage",
    )
    parser.add_argument(
        "--max_n_modes",
        type=int,
        default=None,
        help="max_n_modes, for padding purposes",
    )

    # ============================= PRETRAIN ======================================
    parser.add_argument(
        "--pretrain",
        default=False,
        action="store_true",
        help="pretrain with ortools",
    )
    parser.add_argument(
        "--pretrain_prob",
        type=float,
        default=0.9,
        help="target prob for or tools action",
    )
    parser.add_argument(
        "--pretrain_dataset_generation",
        default="online",
        choices=["online", "offline"],
    )
    parser.add_argument(
        "--pretrain_weight_decay",
        type=float,
        default=1e-1,
        help="pretrain weight decay",
    )
    parser.add_argument(
        "--pretrain_num_envs",
        type=int,
        default=100,
        help="number of pretrain envs (1 is enough for determinisitic case)",
    )
    parser.add_argument(
        "--pretrain_num_eval_envs",
        type=int,
        default=10,
        help="number of pretrain envs (1 is enough for determinisitic case)",
    )
    parser.add_argument(
        "--pretrain_trajectories",
        type=int,
        default=10,
        help="number of trajectories sampled per envs",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=10,
        help="number of pretrain epochs",
    )
    parser.add_argument(
        "--pretrain_batch_size",
        type=int,
        default=128,
        help="size of batch_size for pretrain",
    )
    parser.add_argument(
        "--pretrain_lr",
        type=float,
        default=2e-4,
        help="learning rate for pretrain",
    )
    parser.add_argument(
        "--pretrain_vf_coef",
        type=float,
        default=0,
        help="value function loss weight (set to 0 to deactivate)",
    )

    # =================================================OTHER====================================================================
    parser.add_argument(
        "--max_shared_mem_per_worker",
        default=2000000,
        type=int,
        help="max shared memory per worker",
    )

    parser.add_argument(
        "--skip_initial_eval",
        default=False,
        action="store_true",
        help="Do not perform initial eval",
    )
    parser.add_argument(
        "--skip_model_trace",
        default=False,
        action="store_true",
        help="Do not print the model trace (torchinfo)",
    )

    parser.add_argument(
        "--display_gantt",
        default=False,
        action="store_true",
        help="display gantt-like execution",
    )

    parser.add_argument(
        "--taillard_pbs",
        help="taillard problem name (e.g ta01), default is empty for benchmarking all problems",
        default="*",
    )
    parser.add_argument(
        "--load_problem",
        type=str,
        default=None,
        help="Load problem in Taillard format (machine numbering starts at 0)",
    )
    parser.add_argument("--train_dir", type=str, default=None, help="psp train dir")
    parser.add_argument("--test_dir", type=str, default=None, help="psp test dir")
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.2,
        help="train/test split if no test_dir is provided",
    )

    parser.add_argument(
        "--first_machine_id_is_one",
        default=False,
        action="store_true",
        help="in taillard format, first machine id is 1",
    )
    parser.add_argument(
        "--load_from_job", type=int, default=0, help="Start load at job n from problem"
    )
    parser.add_argument(
        "--load_max_jobs", type=int, default=-1, help="Load at most n jobs from problem"
    )
    parser.add_argument(
        "--sample_n_jobs",
        type=int,
        default=-1,
        help="Sample n jobs from problem during reset",
    )
    parser.add_argument(
        "--chunk_n_jobs",
        type=int,
        default=-1,
        help="Pick a chunk of n jobs from problem during reset",
    )
    parser.add_argument(
        "--validate_on_total_data",
        default=False,
        action="store_true",
        help="set to do validation on total data and not on sample_n_jobs sampled jobs",
    )
    parser.add_argument(
        "--generate_duration_bounds",
        type=float,
        nargs=2,
        default=None,
        help="Generate duration bounds in %% of the true value, e.g. 0.05 0.1 for lower bounds 5%% below loaded value and for upper bounds  10%% above loaded value",
    )
    parser.add_argument(
        "--scaling_constant_ortools",
        type=int,
        default=1000,
        help="Factor for OR-Tools, since it only solves integer problems",
    )
    parser.add_argument(
        "--disable_visdom",
        action="store_true",
        help="Disable visdom logging",
    )
    parser.add_argument(
        "--disable_ortools",
        action="store_true",
        help="Disable ortools solution computation",
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> Tuple[argparse.Namespace, str, str]:
    # ================================================PARSING, IMPORTS, AND VERIFICATIONS=======================================
    # Parsing
    args = parser.parse_args()
    exp_name = get_exp_name(args)
    # path = get_path(args.path, exp_name)

    if args.eval_n_j is None:
        args.eval_n_j = args.n_j

    if args.eval_n_m is None:
        args.eval_n_m = args.n_m

    # Max n_jobs must be under n_jobs
    if args.max_n_j == -1:
        args.max_n_j = max(args.n_j, args.eval_n_j)
    elif args.max_n_j < args.n_j:
        raise Exception(
            "Max number of jobs should be higher than current number of jobs"
        )

    # Max n_machines must be under n_machines
    if args.max_n_m == -1:
        args.max_n_m = max(args.n_m, args.eval_n_m)
    elif args.max_n_m < args.n_m:
        raise Exception(
            "Max number of machines should be higher than current number of machines"
        )

    # Incompatible options
    if args.fixed_random_validation and not args.fixed_validation:
        raise Exception("--fixed_random_validation requires --fixed_validation")
    if args.sample_n_jobs != -1 and args.chunk_n_jobs != -1:
        raise Exception("--sample_n_jobs and --chunk_n_jobs are incompatible")

    # Sorting the features
    args.features = sorted(args.features)

    if args.custom_heuristic_names is None:
        args.custom_heuristic_names = []

    if args.ortools_strategy is None:
        args.ortools_strategy = ["averagistic"]

    if args.resume:
        args.skip_initial_eval = True

    return args, exp_name
