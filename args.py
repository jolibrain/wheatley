import argparse

parser = argparse.ArgumentParser(
    description="Arguments for the main experiment, using PPO to solve a Job Shop Scheduling Problem"
)

# General problem arguments
parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
parser.add_argument("--transition_model_config", type=str, default="L2D", help="Which transition model to use")
parser.add_argument("--reward_model_config", type=str, default="L2D", help="Which reward model to use")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

parser.add_argument(
    "--remove_machine_id", default=False, action="store_true", help="Remove the machine id from the node embedding"
)
parser.add_argument("--fixed_benchmark", default=False, action="store_true", help="Test model on fixed or random benchmark")

# Training arguments
parser.add_argument("--total_timesteps", type=int, default=int(1e4), help="Number of training env timesteps")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for updating the PPO parameters")
parser.add_argument("--n_steps_episode", type=int, default=256, help="Number of steps per episode.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training of PPO")
parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter")
parser.add_argument("--ent_coef", type=float, default=0.005, help="Entropy coefficient")
parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use")

parser.add_argument("--n_test_env", type=int, default=50, help="Number of testing environments during traing")
parser.add_argument("--eval_freq", type=int, default=1000, help="Number of steps between each evaluation during training")

parser.add_argument(
    "--dont_normalize_input", default=False, action="store_true", help="Default is dividing input by constant"
)
parser.add_argument("--fixed_problem", default=False, action="store_true", help="Fix affectations and durations for train")

parser.add_argument("--n_workers", type=int, default=1, help="Number of CPU cores for simulating environment")
parser.add_argument("--multiprocessing", default=False, action="store_true", help="Wether to use multiprocessing or not")

parser.add_argument(
    "--retrain",
    default=False,
    action="store_true",
    help="If true, the script checks for already existing model and use it as a basis for training",
)

# Testing arguments
parser.add_argument("--n_test_problems", type=int, default=100, help="Number of problems for testing")

# Other
parser.add_argument("--exp_name_appendix", type=str, help="Appendix for the name of the experience")
parser.add_argument("--stable_baselines3_localisation", type=str, help="If using custom SB3, specify here the path")

# Parsing
args = parser.parse_args()

# Experience name
exp_name = f"{args.n_j}j{args.n_m}m_{args.seed}seed_{args.transition_model_config}_{args.reward_model_config}"
if args.remove_machine_id:
    exp_name += "_RMI"
if args.fixed_benchmark:
    exp_name += "_FB"
if args.dont_normalize_input:
    exp_name += "_DNI"
if args.fixed_problem:
    exp_name += "_FP"
if args.exp_name_appendix is not None:
    exp_name += "_" + args.exp_name_appendix

# Modify path if there is a custom SB3 library path specified
if args.stable_baselines3_localisation is not None:
    import sys

    sys.path.insert(0, args.stable_baselines3_localisation + "/stable-baselines3/")
    sys.path.insert(0, args.stable_baselines3_localisation + "stable-baselines3/")
    sys.path.insert(0, args.stable_baselines3_localisation)
    import stable_baselines3

    print(f"Stable Baselines 3 imported from : {stable_baselines3.__file__}")
