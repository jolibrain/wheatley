import argparse

parser = argparse.ArgumentParser(
    description="Arguments for the main experiment, using PPO to solve a Job Shop Scheduling Problem"
)

# General problem arguments
parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--path",
    type=str,
    default="saved_networks/default_net",
    help="Path to saved model",
)
parser.add_argument(
    "--add_machine_id",
    default=False,
    action="store_true",
    help="Add the machine id in the node embedding",
)

# Training arguments
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=int(1e4),
    help="Number of training timesteps (corresponding to env timesteps)",
)
parser.add_argument(
    "--n_test_env",
    type=int,
    default=5,
    help="Number of testing environments during traing",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=200,
    help="Evaluation of model is made every eval_freq timesteps",
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=1,
    help="Number of epochs for updating the PPO parameters",
)
parser.add_argument(
    "--n_steps_episode",
    type=int,
    default=256,
    help="Number of steps per episode. This should be bigger than the average number of steps in 1 env run.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size during training of PPO",
)
parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
parser.add_argument(
    "--clip_range", type=float, default=0.2, help="Clipping parameter"
)
parser.add_argument(
    "--ent_coef", type=float, default=0.005, help="Entropy coefficient"
)
parser.add_argument(
    "--vf_coef", type=float, default=0.5, help="Value function coefficient"
)
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument(
    "--dont_divide_loss",
    default=False,
    action="store_true",
    help="Remove the dividing loss hack",
)
parser.add_argument(
    "--fixed_problem",
    default=False,
    action="store_true",
    help="Fix affectations and durations",
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=1,
    help="Number of CPU cores for simulating environment",
)
parser.add_argument(
    "--multiprocessing",
    default=False,
    action="store_true",
    help="Wether to use multiprocessing or not",
)

# Testing arguments
parser.add_argument(
    "--n_test_problems",
    type=int,
    default=100,
    help="Number of problems for testing",
)

# Debug options
parser.add_argument(
    "--fix_problem_size",
    default=False,
    action="store_true",
    help="Wether the size of the problem (n_jobs, n_machines) is fixed for the agent or not",
)
parser.add_argument(
    "--remove_mask",
    default=False,
    action="store_true",
    help="Removes the mask, to check wheter there is a problem in its implementation",
)

# Parsing
args = parser.parse_args()
