"""Evaluate trained model on all instances size."""
import argparse
from pathlib import Path

from tqdm import tqdm

from generic.agent_validator import AgentValidator
from generic.training_specification import TrainingSpecification
from jssp.description import Description
from jssp.env.env import Env
from jssp.env.env_specification import EnvSpecification
from jssp.models.agent import Agent


def load_env(args: argparse.Namespace, n_j: int, n_m: int) -> Env:
    env_specification = EnvSpecification(
        max_n_jobs=n_j,
        max_n_machines=n_m,
        normalize_input=not args.dont_normalize_input,
        input_list=args.features,
        insertion_mode=args.insertion_mode,
        max_edges_factor=args.max_edges_upper_bound_factor,
        sample_n_jobs=args.sample_n_jobs,
        chunk_n_jobs=args.chunk_n_jobs,
        observe_conflicts_as_cliques=args.conflicts == "clique"
        and args.precompute_cliques,
        observe_real_duration_when_affect=args.observe_duration_when_affect,
        do_not_observe_updated_bounds=args.do_not_observe_updated_bounds,
    )
    problem_description = Description(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.reward_model_config,
        deterministic=(args.duration_type == "deterministic"),
        fixed=args.fixed_problem,
        seed=args.seed,
        affectations=None,
        durations=None,
        n_jobs=n_j,
        n_machines=n_m,
        max_duration=args.max_duration,
        duration_mode_bounds=args.duration_mode_bounds,
        duration_delta=args.duration_delta,
    )
    return Env(problem_description, env_specification)


def generate_instances(instances: list, args: argparse.Namespace) -> dict:
    root_dir = Path(f"./instances/jssp/{args.duration_type}")
    root_dir.mkdir(parents=True, exist_ok=True)

    for n_j, n_m in instances:
        env = load_env(args, n_j, n_m)
        instances_dir = root_dir / f"{n_j}x{n_m}/"
        instances_dir.mkdir(exist_ok=True)

        for instance_id in tqdm(range(args.n_validation_env), desc=f"Generating {n_j}x{n_m}"):
            env.reset(soft=False)
            env.state.save_instance_file(
                instances_dir / f"{n_j}x{n_m}_{instance_id}.npz"
            )


if __name__ == "__main__":
    from args import argument_parser, parse_args

    parser = argument_parser()
    args, _, _ = parse_args(parser)

    instances = [
        (6, 6),
        (10, 10),
        (15, 15),
        (20, 20),
        (30, 10),
        (60, 10),
        (20, 15),
        (30, 15),
        (30, 20),
        (50, 15),
        (50, 20),
        (100, 20),
    ]

    generate_instances(instances, args)
