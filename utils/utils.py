import numpy as np


def generate_problem(n_jobs, n_machines, high):
    """
    Generate a random intance of a JSS problem, of size (n_jobs, n_machines),
    with times comprised between specified lower and higher bound
    """
    durations = np.random.randint(low=1, high=high, size=(n_jobs, n_machines))
    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_machines, axis=0)
    affectations = permute_rows(affectations)
    return affectations, durations


def permute_rows(x):
    """
    x is a bidimensional numpy array
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).transpose()
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def node_to_job_and_task(node_id, n_machines):
    return node_id // n_machines, node_id % n_machines


def job_and_task_to_node(job_id, task_id, n_machines):
    return job_id * n_machines + task_id
