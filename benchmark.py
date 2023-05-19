import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from torch.cuda import OutOfMemoryError
from tqdm import tqdm

from args import get_exp_name, get_path
from instances.generate_taillard import generate_taillard, taillard_to_str
from train import main


def sample_hyperparams(hyperparams: Dict[str, list]) -> Dict[str, Any]:
    sampled = dict()
    for key, values in hyperparams.items():
        sampled[key] = random.choice(values)
    return sampled


def log_hyperparams(hyperparams: Dict[str, Any], logfile: Path):
    # Save the hyperparams in a json file.
    with open(logfile, "w") as f:
        json.dump(hyperparams, f, indent=4)


def benchmark_seed(args, n_tries: int):
    base_args = deepcopy(args)

    for seed in tqdm(range(n_tries), desc="BENCHMARKING SEED"):
        args = deepcopy(base_args)

        # Set specific args.
        setattr(args, "seed", seed)
        setattr(args, "exp_name_appendix", f"seed-{seed}")
        exp_name = get_exp_name(args)
        path = get_path(args.path, exp_name)

        main(args, exp_name, path)


def benchmark_single_experiment_generalisation(args, n_tries: int):
    base_args = deepcopy(args)
    for seed in tqdm(range(n_tries), desc="BENCHMARKING SINGLE EXP GENERALISATION"):
        args = deepcopy(base_args)

        # Generate a training instance.
        n_m, n_j = args.n_m, args.n_j
        taillard = generate_taillard(n_j, n_m, seed=seed)
        taillard = taillard_to_str(taillard)
        filename = f"{n_j}x{n_m}-{seed}.txt"
        instance_path = f"instances/generated/{filename}"
        with open(instance_path, "w") as f:
            f.write(taillard)

        # Set specific args.
        setattr(args, "seed", seed)
        setattr(args, "exp_name_appendix", f"single-exp-generalisation-{seed}")
        setattr(args, "load_problem", instance_path)
        setattr(args, "first_machine_id_is_one", True)
        setattr(args, "fixed_validation", True)
        setattr(args, "n_validation_env", 5)

        exp_name = get_exp_name(args)
        path = get_path(args.path, exp_name)

        main(args, exp_name, path)


def benchmark_small_exp_to_big_exp_generalisation(args, n_tries: int):
    base_args = deepcopy(args)
    for seed in tqdm(
        range(n_tries), desc="BENCHMARKING SMALL EXP TO BIG GENERALISATION"
    ):
        args = deepcopy(base_args)

        # Generate a training instance.
        n_m, n_j = args.n_m, args.n_j
        taillard = generate_taillard(n_j, n_m, seed=seed)
        taillard = taillard_to_str(taillard)
        filename = f"{n_j}x{n_m}-{seed}.txt"
        instance_path = f"instances/generated/{filename}"
        with open(instance_path, "w") as f:
            f.write(taillard)

        # Set specific args.
        setattr(args, "seed", seed)
        setattr(args, "exp_name_appendix", f"single-exp-generalisation-{seed}")
        setattr(args, "load_problem", instance_path)
        setattr(args, "first_machine_id_is_one", True)
        setattr(args, "fixed_validation", True)
        setattr(args, "n_validation_env", 5)

        exp_name = get_exp_name(args)
        path = get_path(args.path, exp_name)

        main(args, exp_name, path)


def benchmark_dgl_hyperparams(args, n_tries: int):
    params_space = {
        "graph_pooling": ["max", "learn"],
        "layer_pooling": ["last", "all"],
        "mlp_act_graph": ["relu", "tanh", "gelu", "selu"],
        "n_mlp_layers_features_extractor": [1, 3, 5],
        "n_layers_features_extractor": [1, 3, 6, 10],
        "hidden_dim_features_extractor": [16, 32, 64],
        "residual_gnn": [True, False],
        "normalize_gnn": [True, False],
        "fe_type": ["dgl"],
        "n_mlp_layers_actor": [1],
        "n_mlp_layers_critic": [1],
        "hidden_dim_actor": [32],
        "hidden_dim_critic": [32],
        "graph_has_relu": [True],
    }

    for seed in tqdm(range(n_tries), desc="BENCHMARKING DGL HYPERPARAMS"):
        args = deepcopy(args)

        setattr(args, "seed", seed)

        # Sample hyperparams.
        sampled = sample_hyperparams(params_space)
        for key, value in sampled.items():
            setattr(args, key, value)

        # Set specific args.
        setattr(args, "exp_name_appendix", f"dgl-hyperparams_{seed}")
        exp_name = get_exp_name(args)
        path = get_path(args.path, exp_name)

        # Disable visdom to avoid spamming the server.
        setattr(args, "disable_visdom", True)

        try:
            main(args, exp_name, path)
        except OutOfMemoryError:
            print("Out of memory, skipping.")
            sampled["out_of_memory"] = True
        finally:
            log_hyperparams(sampled, Path(path) / "hyperparams.json")


if __name__ == "__main__":
    from args import args

    # benchmark_seed(args, 10)
    # benchmark_single_experiment_generalisation(args, 10)
    benchmark_dgl_hyperparams(args, 100)
