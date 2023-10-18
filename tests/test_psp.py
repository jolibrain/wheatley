import sys

import pytest

from args import argument_parser, parse_args
from psp.train_psp import main

# This is the list of all experiments we want to try.
# Each entry should be read as "argument_name: [value_experiment_1, value_experiment_2, ...]"
# If an entry has less experiment values than the maximum number of experiments,
# then its last value will be repeated for the missing experiment values.
possible_args = {
    "n_j": [6, 10],
    "n_m": [10, 5],
    "dont_normalize_input": [True, False],
    # "sample_n_jobs": [-1, 3],  # WARNING: This option does not work for PSPs?
    "chunk_n_jobs": [3, -1],
    "max_n_j": [20],
    "max_n_m": [20],
    "lr": [0.0001],
    "ent_coef": [0.05],
    "vf_coef": [1.0],
    "clip_range": [0.25],
    "gamma": [1.0],
    "gae_lambda": [0.99],
    "optimizer": ["adamw"],
    "fe_type": ["dgl"],
    "residual_gnn": [True],
    "graph_has_relu": [True],
    "graph_pooling": ["learn"],
    "hidden_dim_features_extractor": [32],
    "n_layers_features_extractor": [3],
    "mlp_act": ["gelu"],
    "layer_pooling": ["last"],
    "n_mlp_layers_features_extractor": [1],
    "n_mlp_layers_actor": [1],
    "n_mlp_layers_critic": [1],
    "hidden_dim_actor": [16],
    "hidden_dim_critic": [16],
    "total_timesteps": [1000],
    "n_validation_env": [10],
    "n_steps_episode": [490],
    "batch_size": [245],
    "n_epochs": [1],
    "fixed_validation": [True],
    # "custom_heuristic_names": ["SPT MWKR MOPNR FDD/MWKR", "SPT"],
    "seed": [0],
    "duration_type": ["stochastic", "deterministic"],
    "generate_duration_bounds": ["0.05 0.1"],
    "ortools_strategy": ["averagistic realistic", "pessimistic"],
    "device": ["cuda:0"],
    "n_workers": [1, 2],
    "skip_initial_eval": [True, False],
    "exp_name_appendix": ["test"],
    "train_dir": ["./instances/psp/test/", "./instances/psp/test/"],
}

# Duplicate each entry to match the maximum number of possibilities to try.
max_number_of_possibilities = max(len(v) for v in possible_args.values())
possible_args = {
    k: v + (max_number_of_possibilities - len(v)) * [v[-1]]
    for k, v in possible_args.items()
}

# Build the list of all experiments to launch.
# An experiment is defined by a list of arguments preformatted
# for argparse.
args_to_test = [[] for _ in range(max_number_of_possibilities)]
for key, values in possible_args.items():
    for test_id, value in enumerate(values):
        args = args_to_test[test_id]

        if value is True:
            args.append(f"--{key}")
        elif value is False:
            pass
        else:
            args.append(f"--{key}")
            if isinstance(value, str):
                for sub_v in value.split(" "):
                    args.append(f"{sub_v}")
                    print(sub_v)
            else:
                args.append(f"{value}")


@pytest.mark.parametrize(
    "args",
    args_to_test,
)
def test_args(args: list):
    """Make sure the main training function don't crash when using multiple different
    args.
    """
    original_argv = sys.argv

    try:
        # Simulate the arguments passed as input for the argument parser to work seemlessly.
        args.append("--disable_visdom")
        if len(original_argv) == 3:
            output_directory = original_argv[2]
            output_directory += "/" if not output_directory.endswith("/") else ""
            args.append("--path")
            args.append(output_directory)

        sys.argv = ["python3"] + args

        parser = argument_parser()
        args, exp_name, path = parse_args(parser)
        main(args, exp_name, path)
    finally:
        # Don't forget to bring back the old argv!
        sys.argv = original_argv
