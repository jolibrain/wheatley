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
    np.save(f"benchmark/generated_data{n_jobs}_{n_machines}_seed200.npy", data)
    return data

def load_taillard_problem(problem_file, taillard_offset=True):
    # http://jobshop.jjvh.nl/explanation.php#taillard_def
    with open(problem_file, 'r') as f:

        line = next(f)
        while line[0] == '#':
            line = next(f)
        
        # header
        header = line
        head_list = [int(i) for i in header.split()]
        assert(len(head_list) == 2)
        n_j = head_list[0]
        n_m = head_list[1]

        line = next(f)
        while line[0] == '#':
            line = next(f)
        
        # matrix of durations
        np_lines = []
        for j in range(n_j):
            dur_list = [float(i) for i in line.split()]
            np_lines.append(np.array(dur_list))
            line = next(f)
        durations = np.stack(np_lines)

        while line[0] == '#':
            line = next(f)
        
        # matrix of affectations
        if taillard_offset:
            toffset = 1
        else:
            toffset = 0
        np_lines = []
        for j in range(n_j):
            aff_list = [int(i)-toffset for i in line.split()]  # Taillard spec has machines id start at 1
            np_lines.append(np.array(aff_list))
            line = next(f,'')
            if line == '':
                break
        affectations = np.stack(np_lines)

        return n_j, n_m, affectations, durations
