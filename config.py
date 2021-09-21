import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
N_MLP_LAYERS_FEATURES_EXTRACTOR = 4
N_LAYERS_FEATURES_EXTRACTOR = 2
HIDDEN_DIM_FEATURES_EXTRACTOR = 64
N_MLP_LAYERS_ACTOR = 4
HIDDEN_DIM_ACTOR = 32
N_MLP_LAYERS_CRITIC = 4
HIDDEN_DIM_CRITIC = 32

MAX_DURATION = 99  # Handle this with care, linked to a lot of other things

# Parameters that shouldn't play a role in learning
MAX_N_JOBS = 5
MAX_N_MACHINES = 5

MAX_N_NODES = MAX_N_JOBS * MAX_N_MACHINES
MAX_N_EDGES = MAX_N_NODES ** 2
