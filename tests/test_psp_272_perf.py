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
                "2e-4",
                "--ent_coef",
                "0.005",
                "--vf_coef",
                "0.5",
                "--target_kl",
                "0.04",
                "--clip_range",
                "0.25",
                "--gamma",
                "1.0",
                "--gae_lambda",
                "1.0",
                "--optimizer",
                "adam",
                "--fe_type",
                "dgl",
                "--graph_pooling",
                "learn",
                "--hidden_dim_features_extractor",
                "64",
                "--n_layers_features_extractor",
                "6",
                "--layer_pooling",
                "all",
                "--n_mlp_layers_features_extractor",
                "3",
                "--n_mlp_layers_actor",
                "1",
                "--n_mlp_layers_critic",
                "1",
                "--hidden_dim_actor",
                "64",
                "--hidden_dim_critic",
                "64",
                "--total_timesteps",
                "100_000",
                "--n_validation_env",
                "1",
                "--n_steps_episode",
                "1800",
                "--batch_size",
                "1024",
                "--n_epochs",
                "20",
                "--fixed_validation",
                "--device",
                "cuda",
                "--exp_name",
                "272-psp",
                "--load_problem",
                "./instances/psp/272/272.sm",
                "--n_workers",
                "10",
                "--weight_decay",
                "0.0",
                "--skip_initial_eval",
            ],
            1.0,
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
