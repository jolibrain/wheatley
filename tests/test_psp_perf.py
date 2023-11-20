import sys

import pytest

from args import argument_parser, parse_args
from psp.train_psp import main


@pytest.mark.parametrize(
    "args, upper_limit_ratios",
    [
        (
            [
                "--lr",
                "1e-4",
                "--ent_coef",
                "0.05",
                "--vf_coef",
                "2.0",
                "--target_kl",
                "0.04",
                "--clip_range",
                "0.20",
                "--gamma",
                "1.00",
                "--gae_lambda",
                "0.99",
                "--optimizer",
                "adamw",
                "--fe_type",
                "dgl",
                "--residual_gnn",
                "--graph_has_relu",
                "--graph_pooling",
                "max",
                "--hidden_dim_features_extractor",
                "16",
                "--n_layers_features_extractor",
                "5",
                "--mlp_act",
                "gelu",
                "--layer_pooling",
                "last",
                "--n_mlp_layers_features_extractor",
                "1",
                "--n_mlp_layers_actor",
                "1",
                "--n_mlp_layers_critic",
                "1",
                "--hidden_dim_actor",
                "16",
                "--hidden_dim_critic",
                "16",
                "--total_timesteps",
                "100_000",
                "--n_validation_env",
                "1",
                "--n_steps_episode",
                "2450",
                "--batch_size",
                "128",
                "--n_epochs",
                "3",
                "--fixed_validation",
                "--seed",
                "0",
                "--duration_type",
                "stochastic",
                "--ortools_strategy",
                "averagistic",
                "--generate_duration_bounds",
                "0.05",
                "0.1",
                "--device",
                "cuda",
                "--exp_name",
                "small-psp",
                "--load_problem",
                "instances/psp/sm/j30/j3010_1.sm",
                "--n_workers",
                "1",
            ],
            1.10,
        ),
    ],
)
def test_performance(args: list, upper_limit_ratios: float):
    """Make sure PPO is doing OK on some trainings."""
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
        ppo_ratio = main(args, exp_name, path)

        assert (
            ppo_ratio <= upper_limit_ratios
        ), f"Training is under-performing ({ppo_ratio} vs {upper_limit_ratios})"
    finally:
        sys.argv = original_argv
