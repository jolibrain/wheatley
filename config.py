import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
N_MLP_LAYERS_FEATURES_EXTRACTOR = 3
N_LAYERS_FEATURES_EXTRACTOR = 3
INPUT_DIM_FEATURES_EXTRACTOR = 2
HIDDEN_DIM_FEATURES_EXTRACTOR = 32
N_MLP_LAYERS_ACTOR = 3
HIDDEN_DIM_ACTOR = 32
N_MLP_LAYERS_CRITIC = 3
HIDDEN_DIM_CRITIC = 32

# Parameters that shouldn't play a role in learning
MAX_N_JOBS = 10
MAX_N_MACHINES = 10

MAX_N_NODES = MAX_N_JOBS * MAX_N_MACHINES
MAX_N_EDGES = MAX_N_NODES ** 2
