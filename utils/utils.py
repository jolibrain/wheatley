from os import path

import numpy as np
import torch

from config import MAX_N_MACHINES, MAX_DURATION


def generate_problem(n_jobs, n_machines, high):
    """
    Generate a random intance of a JSS problem, of size (n_jobs, n_machines),
    with times comprised between specified lower and higher bound
    """
    durations = np.random.randint(low=1, high=high, size=(n_jobs, n_machines))
    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_jobs, axis=0)
    affectations = _permute_rows(affectations)
    return affectations, durations


def _permute_rows(x):
    """
    x is a bidimensional numpy array
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).transpose()
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def generate_data(n_j, n_m, max_duration, seed=200, n_problems=100):
    np.random.seed(seed)
    data = np.array([generate_problem(n_j, n_m, max_duration) for _ in range(n_problems)])
    return data


def node_to_job_and_task(node_id, n_machines):
    return node_id // n_machines, node_id % n_machines


def job_and_task_to_node(job_id, task_id, n_machines):
    return job_id * n_machines + task_id


def load_benchmark(n_jobs, n_machines):
    if not path.exists(f"benchmark/generated_data{n_jobs}_{n_machines}_seed200.npy"):
        data = generate_data(n_jobs, n_machines, MAX_DURATION)
    else:
        data = np.load(f"benchmark/generated_data{n_jobs}_{n_machines}_seed200.npy")
    np.save(f"generated_data{n_jobs}_{n_machines}_seed200.npy", data)
    return data
