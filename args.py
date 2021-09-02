import argparse

parser = argparse.ArgumentParser(
    description="Arguments for the main experiment, using PPO to solve a Job Shop Scheduling Problem"
)

# General problem arguments
parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
parser.add_argument(
    "--n_j_testing", type=int, default=5, help="Number of jobs for test"
)
parser.add_argument(
    "--n_m_testing", type=int, default=5, help="Number of jobs for test"
)

parser.add_argument(
    "--n_timesteps",
    type=int,
    default=int(1e4),
    help="Number of training timesteps (corresponding to env timesteps)",
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=1,
    help="Number of epochs for updating the PPO parameters",
)
parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
parser.add_argument(
    "--clip_range", type=float, default=0.2, help="Clipping parameter"
)
parser.add_argument(
    "--ent_coef", type=float, default=1, help="Entropy coefficient"
)
parser.add_argument(
    "--vf_coef", type=float, default=0.01, help="Value function coefficient"
)
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

parser.add_argument(
    "--path", type=str, required=True, help="Path to saved model"
)

parser.add_argument(
    "--n_test_problems",
    type=int,
    default=10,
    help="Number of problems for testing",
)

args = parser.parse_args()
