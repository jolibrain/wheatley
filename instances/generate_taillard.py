from typing import Optional

import numpy as np


def generate_taillard(
    n: int, m: int, duration_bounds=None, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a Taillard instance of size n and m
    :param n: Number of jobs.
    :param m: Number of machines.
    :param seed: Random seed.
    :return: A Taillard instance. The first axis is the processing times,
        the second axis is the machine order.
        Shape of [2, n, m].
    """
    rng = np.random.default_rng(seed)

    # Generate the processing times
    if duration_bounds is not None:
        processing_times = rng.integers(
            duration_bounds[0], duration_bounds[1], size=(n, m)
        )
    else:
        processing_times = rng.integers(1, 100, size=(n, m))

    # Generate the machine order
    machines = []
    for _ in range(n):
        # Machines are numbered from 1 to m
        order = rng.permutation(m) + 1
        machines.append(order)
    machines = np.array(machines)

    taillard = np.stack([processing_times, machines], axis=0)
    return taillard


def taillard_to_str(taillard: np.ndarray) -> str:
    desc = ""
    n_jobs, n_machines = taillard.shape[1:]
    desc += f"{n_jobs} {n_machines}\n"

    # Print the processing times.
    for job_id in range(n_jobs):
        processing_times = taillard[0, job_id]
        desc += " ".join(map(str, processing_times)) + "\n"

    # Print the machine order.
    for job_id in range(n_jobs):
        machine_order = taillard[1, job_id]
        desc += " ".join(map(str, machine_order)) + "\n"

    desc = desc.strip()  # Remove the last newline.
    return desc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_j", type=int, default=4)
    parser.add_argument("--n_m", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    taillard = generate_taillard(args.n_j, args.n_m, seed=args.seed)
    desc = taillard_to_str(taillard)
    print(desc)
