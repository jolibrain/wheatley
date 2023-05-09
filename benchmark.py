from copy import deepcopy

from tqdm import tqdm

from args import get_exp_name, get_path
from instances.generate_taillard import generate_taillard, taillard_to_str
from train import main


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


if __name__ == "__main__":
    from args import args

    # benchmark_seed(args, 10)
    benchmark_single_experiment_generalisation(args, 10)
