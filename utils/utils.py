from os import path

import numpy as np
import torch

from config import MAX_N_MACHINES, MAX_DURATION


def generate_problem(n_jobs, n_machines, high):
    """
    Generate a random intance of a JSS problem, of size (n_jobs, n_machines),
    with times comprised between specified lower and higher bound
    """
    durations = np.random.randint(low=1, high=high, size=(n_jobs, n_machines,1))
    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_jobs, axis=0)
    affectations = _permute_rows(affectations)
    return affectations, durations

def generate_problem_distrib(n_jobs, n_machines, duration_mode_bounds, delta_duration):

    durations = np.empty((n_jobs,n_machines,4),dtype = np.int32)
    durations[:,:,3] = np.random.randint(duration_mode_bounds[0],
                                         duration_mode_bounds[1],
                                         size=(n_jobs, n_machines))
    for j in range(n_jobs):
        for m in range(n_machines):
            dd = np.random.randint(1, delta_duration[0])
            durations[j,m][1] = max(1,durations[j,m][3] - dd)
            dd = np.random.randint(1, delta_duration[1])
            durations[j,m][2] = durations[j,m][3] + dd

    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_jobs, axis=0)
    affectations = _permute_rows(affectations)
    return affectations, durations

def generate_problem_durations(durations):
    n_jobs = durations.shape[0]
    n_machines = durations.shape[1]
    for j in range(n_jobs):
        for m in range(n_machines):
            if durations[j,m,1] == durations[j,m,2]:
                durations[j,m,0] = durations[j,m,1]
            else:
                if durations[j,m][1] > durations[j,m][3]:
                    print(durations[j,m][1] , durations[j,m][3])
                durations[j,m,0] = np.random.triangular(durations[j,m][1], durations[j,m][3],
                                                        durations[j,m][2])
    return durations

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

def load_taillard_problem(problem_file, taillard_offset=True, fixed_distrib=False):
    # http://jobshop.jjvh.nl/explanation.php#taillard_def
    if fixed_distrib:
        print("will load problem with uncertainties, using extended taillard format")

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


        if not fixed_distrib:
            durations = np.expand_dims(durations, axis=2)
        else:

            mode_durations = durations

            while line[0] == '#':
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = [float(i) for i in line.split()]
                np_lines.append(np.array(dur_list))
                line = next(f)
            min_durations = np.stack(np_lines)

            while line[0] == '#':
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = [float(i) for i in line.split()]
                np_lines.append(np.array(dur_list))
                line = next(f)
            max_durations = np.stack(np_lines)

            real_durations= np.zeros((n_j, n_m)) - 1

            durations = np.stack([real_durations, min_durations, max_durations, mode_durations],
                                 axis = 2)


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
