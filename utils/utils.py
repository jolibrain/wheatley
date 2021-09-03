import numpy as np
import torch

from config import MAX_N_MACHINES


def generate_problem(n_jobs, n_machines, high):
    """
    Generate a random intance of a JSS problem, of size (n_jobs, n_machines),
    with times comprised between specified lower and higher bound
    """
    durations = np.random.randint(low=1, high=high, size=(n_jobs, n_machines))
    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_machines, axis=0)
    affectations = _permute_rows(affectations)
    return affectations, durations


def _permute_rows(x):
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


def apply_mask(tensor, mask):
    """
    Returns the tensor corresponding to 1 values in the mask, and the corresponding
    indexes. Tensor should be (A, B, C) shaped and mask (A, B) shaped
    """
    indexes = []
    masked_tensors = []

    for i in range(tensor.shape[0]):
        indexes.append((mask[i] == 1).nonzero(as_tuple=True)[0])
        masked_tensors.append(tensor[i][indexes[-1]])
    return torch.stack(masked_tensors), indexes
