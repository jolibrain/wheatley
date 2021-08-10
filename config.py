import argparse
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
N_MLP_LAYERS_FEATURE_EXTRACTOR = 3
N_LAYERS_FEATURE_EXTRACTOR = 3
INPUT_DIM_FEATURE_EXTRACTOR = 2
HIDDEN_DIM_FEATURE_EXTRACTOR = 32
N_MLP_LAYERS_ACTOR = 3
HIDDEN_DIM_ACTOR = 32
N_MLP_LAYERS_CRITIC = 3
HIDDEN_DIM_CRITIC = 32

# Parameters that shouldn't play a role in learning
MAX_N_JOBS = 10
MAX_N_MACHINES = 10

MAX_N_NODES = MAX_N_JOBS * MAX_N_MACHINES
MAX_N_EDGES = MAX_N_NODES ** 2


parser = argparse.ArgumentParser(
    description="Arguments for the main experiment, using PPO to solve a Job Shop Scheduling Problem"
)

# General problem arguments
parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
parser.add_argument(
    "--n_j_testing", type=int, default=8, help="Number of jobs for test"
)
parser.add_argument(
    "--n_m_testing", type=int, default=8, help="Number of jobs for test"
)

args = parser.parse_args()
