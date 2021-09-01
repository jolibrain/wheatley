import argparse

parser = argparse.ArgumentParser(
    description="Arguments for the main experiment, using PPO to solve a Job Shop Scheduling Problem"
)

# General problem arguments
parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
parser.add_argument(
    "--n_timesteps",
    type=int,
    default=int(1e4),
    help="Number of training timesteps (corresponding to env timesteps)",
)
parser.add_argument(
    "--n_j_testing", type=int, default=5, help="Number of jobs for test"
)
parser.add_argument(
    "--n_m_testing", type=int, default=5, help="Number of jobs for test"
)
parser.add_argument(
    "--max_duration", type=int, default=99, help="Max duration for the jobs"
)


args = parser.parse_args()
