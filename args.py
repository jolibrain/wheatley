import argparse

from utils.utils import get_exp_name, get_path


parser = argparse.ArgumentParser(
    description="These args can be used with train.py, test.py and benchmark/run_taillard.py. They specify how the training"
    "(or testing) is going to be performed"
)

# =================================================PROBLEM DESCRIPTION======================================================
parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
parser.add_argument("--max_duration", type=int, default=99, help="Max duration for problems")

# =================================================COMPUTER SPECIFICATION===================================================
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_n_j", type=int, default=-1, help="Max number of jobs (if -1, max_n_j=n_j")
parser.add_argument("--max_n_m", type=int, default=-1, help="Max number of machines (if -1, max_n_m=n_m")
parser.add_argument(
    "--path", type=str, default="saved_networks/default_net", help="Path to saved network (default is set to exp_name)"
)
parser.add_argument("--n_workers", type=int, default=1, help="Number of CPU cores for simulating environment")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    help="Which device to use (cpu, cuda:0, cuda:1...)",
)
parser.add_argument("--exp_name_appendix", type=str, help="Appendix for the name of the experience")
parser.add_argument("--stable_baselines3_localisation", type=str, help="If using custom SB3, specify here the path")
parser.add_argument(
    "--vecenv_type",
    type=str,
    default="subproc",
    choices=["subproc", "dummy"],
    help="Use SubprocEnv or DummyVecEnv in SB3",
)

# =================================================TRAINING SPECIFICATION====================================================
parser.add_argument("--total_timesteps", type=int, default=int(1e4), help="Number of training env timesteps")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for updating the agent's parameters")
parser.add_argument("--n_steps_episode", type=int, default=1024, help="Number of steps per episode.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training of the agent")
parser.add_argument("--lr", type=float, default=1e-3, help="Default Learning rate")
parser.add_argument("--fe_lr", type=float, default=None, help="Learning rate for feature extractor")
parser.add_argument("--rpo", default=False, action="store_true", help="use RPO-style smoothing")
parser.add_argument("--rpo_smoothing_param", type=float, default=1.0, help="RPO-style smoothing param")

parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    choices=["adam", "sgd", "adamw", "radam", "dadam"],
    help="Which optimizer to use",
)
parser.add_argument("--freeze_graph", default=False, action="store_true", help="Freezes graph during training")
parser.add_argument(
    "--custom_heuristic_name", default="None", choices=["None", "mwkr", "mopnr"], help="Which custom heuristic to run"
)
parser.add_argument(
    "--retrain",
    default=False,
    action="store_true",
    help="Use this flag if you want to retrain a pretrained model. The script will look for existing model at path loc.",
)

# =================================================VALIDATION SPECIFICATION=================================================
parser.add_argument("--n_validation_env", type=int, default=20, help="Number of validation environments ")
parser.add_argument(
    "--fixed_validation", action="store_true", help="Use the same problems/durations sampling and OR-Tools solutions"
)
parser.add_argument(
    "--fixed_random_validation",
    type=int,
    default=0,
    help="Average the random solutions over N random runs, requires --fixed_validation",
)
parser.add_argument("--validation_freq", type=int, default=-1, help="Number of steps between each evaluation")
parser.add_argument("--max_time_ortools", type=int, default=3, help="Max compute time for ortools (in seconds)")
parser.add_argument("--validation_batch_size", type=int, default=0, help="Batch size for predictions of actions")

# =================================================TESTING SPECIFICATION====================================================
parser.add_argument("--n_test_problems", type=int, default=100, help="Number of problems for testing")
parser.add_argument("--test_print_every", type=int, default=50, help="Print frequency for testing")

# =================================================AGENT SPECIFICATION======================================================
parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter")
parser.add_argument("--target_kl", type=float, default=0.2, help="Limit the KL divergence between updates")
parser.add_argument("--ent_coef", type=float, default=0.005, help="Entropy coefficient")
parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
parser.add_argument("--dont_normalize_advantage", action="store_true", help="Whether to not normalize PPO advantage")

parser.add_argument(
    "--gconv_type",
    type=str,
    default="gatv2",
    choices=["gin", "gatv2", "pna", "dgn", "gcn2"],
    help="Graph convolutional neural network type: gin for GIN, gatv2 for GATV2",
)
parser.add_argument(
    "--graph_pooling",
    type=str,
    default="learn",
    choices=["max", "average", "learn"],
    help="which pooling to use (avg , max or learn)",
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
    default="tanh",
    choices=["relu", "tanh", "elu", "gelu", "selu"],
    help="agent mlp extractor activation type",
)
parser.add_argument(
    "--mlp_act",
    type=str,
    default="tanh",
    choices=["relu", "tanh", "elu", "gelu", "selu"],
    help="agent mlp extractor activation type",
)
parser.add_argument("--dropout", type=float, default=0.0, help="dropout ratio")
parser.add_argument(
    "--ortools_strategy",
    type=str,
    default="pessimistic",
    choices=["realistic", "optimistic", "pessimistic", "averagistic"],
    help="ortools durations estimations in pessimistic|optimistic|averagistic|realistic realistic means omiscient, "
    "ie sees the future",
)
parser.add_argument(
    "--fe_type", type=str, default="pyg", help="feature extractor type in [pyg|dgl]", choices=["pyg", "dgl", "tokengt"]
)
parser.add_argument(
    "--transformer_flavor",
    type=str,
    default="vanilla",
    help="transformer implementation for tokengt",
    choices=["vanilla", "linear", "performer"],
)
parser.add_argument("--cache_lap_node_id", action="store_true", help="enable laplacian cache for tokengt")
parser.add_argument("--lap_node_id_k", type=int, default=10, help="laplacian id size for tokengt")
parser.add_argument(
    "--graph_has_relu", action="store_true", help="whether graph feature extractor has activations between layers"
)
parser.add_argument("--n_mlp_layers_features_extractor", type=int, default=2, help="Number of MLP layers in each GNN")
parser.add_argument("--n_layers_features_extractor", type=int, default=4, help="Number of layers of GNN")
parser.add_argument("--hidden_dim_features_extractor", type=int, default=64, help="Dimension of hidden and output for GNN")
parser.add_argument("--n_attention_heads", type=int, default=4, help="Dimension of hidden and output for GNN")
parser.add_argument("--reverse_adj_in_gnn", action="store_true", help="reverse adj matrix in GNN")
parser.add_argument("--residual_gnn", action="store_true", help="use residual connection in GNN")
parser.add_argument("--normalize_gnn", action="store_true", help="normalize gnn everywhere")
parser.add_argument(
    "--conflicts",
    type=str,
    help="machine conflcit encoding in [att|clique|node]",
    default="att",
    choices=["att", "clique", "node"],
)
parser.add_argument(
    "--n_mlp_layers_shared", type=int, default=0, help="Number of MLP layers in shared (excluding input and output"
)
parser.add_argument("--hidden_dim_shared", type=int, default=32, help="Hidden dim for shared")
parser.add_argument(
    "--n_mlp_layers_actor", type=int, default=2, help="Number of MLP layers in actor (excluding input and output"
)
parser.add_argument("--hidden_dim_actor", type=int, default=32, help="Hidden dim for actor")
parser.add_argument(
    "--n_mlp_layers_critic", type=int, default=2, help="Number of MLP layers in critic (excluding input and output)"
)
parser.add_argument("--hidden_dim_critic", type=int, default=32, help="Hidden dim for critic")

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
    default="L2D",
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
    "--reward_model_config",
    type=str,
    default="L2D",
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
    default=(10, 49),
    help="The define the range of sampling for the mode of the triangular distributions for durations",
)
parser.add_argument(
    "--duration_delta",
    type=int,
    nargs=2,
    default=(2, 50),
    help="This defines the delta between low_value/high_value and the mode for the triangular distributions for durations",
)
parser.add_argument(
    "--insertion_mode",
    type=str,
    default="no_forced_insertion",
    choices=["no_forced_insertion", "full_forced_insertion", "choose_forced_insertion", "slot_locking"],
    help="This defines how the jobs are inserted in the schedule.",
)
parser.add_argument(
    "--features",
    type=str,
    nargs="+",
    default=[
        "selectable",
        "duration",
        "total_job_time",
        "total_machine_time",
        "job_completion_percentage",
        "machine_completion_percentage",
        "mopnr",
        "mwkr",
    ],
    help="The features we want to have as input of features_extractor. Should be in {duration, one_hot_job_id, "
    + "one_hot_machine_id, total_job_time, total_machine_time, job_completion_percentage, machine_completion_percentage, "
    + "mopnr, mwkr",
)
parser.add_argument(
    "--dont_normalize_input", default=False, action="store_true", help="Default is dividing input by constant"
)
parser.add_argument("--fixed_problem", default=False, action="store_true", help="Fix affectations and durations")
parser.add_argument(
    "--max_edges_upper_bound_factor",
    type=int,
    default=-1,
    help="Upper bound factor to max_n_edges, allows lowering the overall memory usage",
)

# =================================================OTHER====================================================================
parser.add_argument(
    "--taillard_pbs", help="taillard problem name (e.g ta01), default is empty for benchmarking all problems", default="*"
)
parser.add_argument(
    "--load_problem", type=str, default=None, help="Load problem in Taillard format (machine numbering starts at 0)"
)
parser.add_argument("--load_from_job", type=int, default=0, help="Start load at job n from problem")
parser.add_argument("--load_max_jobs", type=int, default=-1, help="Load at most n jobs from problem")
parser.add_argument("--sample_n_jobs", type=int, default=-1, help="Sample n jobs from problem during reset")
parser.add_argument("--chunk_n_jobs", type=int, default=-1, help="Pick a chunk of n jobs from problem during reset")
parser.add_argument(
    "--validate_on_total_data",
    default=False,
    action="store_true",
    help="set to do validation on total data and not on sample_n_jobs sampled jobs",
)
parser.add_argument(
    "--generate_duration_bounds",
    type=float,
    default=-1.0,
    help="Generate duration bounds in %% of the true value, e.g. 0.05 for 5%% uncertainty",
)
parser.add_argument(
    "--scaling_constant_ortools", type=int, default=1000, help="Factor for OR-Tools, since it only solves integer problems"
)

# ================================================PARSING, IMPORTS, AND VERIFICATIONS=======================================
# Parsing
args = parser.parse_args()
exp_name = get_exp_name(args)
path = get_path(args.path, exp_name)

# Modify path if there is a custom SB3 library path specified
if args.stable_baselines3_localisation is not None:
    import sys

    sys.path.insert(0, args.stable_baselines3_localisation + "/stable-baselines3/")
    sys.path.insert(0, args.stable_baselines3_localisation + "stable-baselines3/")
    sys.path.insert(0, args.stable_baselines3_localisation)
    import stable_baselines3

    print(f"Stable Baselines 3 imported from : {stable_baselines3.__file__}")

# Max n_jobs must be under n_jobs
if args.max_n_j == -1:
    args.max_n_j = args.n_j
elif args.max_n_j < args.n_j:
    raise Exception("Max number of jobs should be higher than current number of jobs")

# Max n_machines must be under n_machines
if args.max_n_m == -1:
    args.max_n_m = args.n_m
elif args.max_n_m < args.n_m:
    raise Exception("Max number of machines should be higher than current number of machines")

# Incompatible options
if args.fixed_random_validation and not args.fixed_validation:
    raise Exception("--fixed_random_validation requires --fixed_validation")
if args.sample_n_jobs != -1 and args.chunk_n_jobs != -1:
    raise Exception("--sample_n_jobs and --chunk_n_jobs are incompatible")

# Sorting the features
args.features = sorted(args.features)
