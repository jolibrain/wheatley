# Parameters
N_MLP_LAYERS_FEATURES_EXTRACTOR = 4
N_LAYERS_FEATURES_EXTRACTOR = 5
HIDDEN_DIM_FEATURES_EXTRACTOR = 64
N_ATTENTION_HEADS = 4
N_MLP_LAYERS_ACTOR = 4
HIDDEN_DIM_ACTOR = 32
N_MLP_LAYERS_CRITIC = 4
HIDDEN_DIM_CRITIC = 32

MAX_DURATION = 99  # Handle this with care, linked to a lot of other things

DURATION_MODE_BOUNDS = (10,49) # durations intrevals are defined by values in 10, 89
DURATION_DELTA = (2,50) # (v1, v2) a random value within [0,v1] is removed from value above
# a random value with [0,v2] is added to value above in roder to get an inveral
# v1 = v2 : intevals are statiscally centered around value drawn from above
# interval mean size is v1+v2/2
# v1 = v2 = 0 : zero size interval=> certainty
# v1 = 0 means values are considered as min
# v2 = 0 means values are considered as max

# Parameters that shouldn't play a role in learning
MAX_N_JOBS = 20
MAX_N_MACHINES = 20

MAX_N_NODES = MAX_N_JOBS * MAX_N_MACHINES
MAX_N_EDGES = MAX_N_NODES ** 2

# OR-Tools parameters
MAX_TIME_ORTOOLS = 1
SCALING_CONSTANT_ORTOOLS=1000000
