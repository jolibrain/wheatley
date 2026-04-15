import numpy as np
import scipy as sp
import math
import torch
from typing import Any, Optional


def laplacian_matrix(
    senders: np.ndarray,
    receivers: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n: Optional[int] = None,
) -> Any:
    """Creates the laplacian matrix for given edge list.
    The edge list should be symmetric, and there should not be any isolated nodes.
    Args:
      senders: The sender nodes of the graph
      receivers: The receiver nodes of the graph
      weights: The weights of the edges
    Returns:
      A sparse Laplacian matrix
    """

    if weights is None:
        weights = 0 * senders + 1

    if n is None:
        n = senders.max()
        if receivers.max() > n:
            n = receivers.max()
        n += 1

    s = senders.tolist() + list(range(n))
    t = receivers.tolist() + list(range(n))
    w = weights.tolist() + [0.0] * n
    adj = sp.sparse.csc_matrix((w, (s, t)), shape=(n, n))
    lap = adj * -1.0
    lap.setdiag(np.ravel(adj.sum(axis=0)))
    return lap


def laplacian_eigenv(
    senders: np.ndarray,
    receivers: np.ndarray,
    weights: Optional[np.ndarray] = None,
    k=2,
    n: Optional[int] = None,
):
    """Computes the k smallest non-trivial eigenvalue and eigenvectors of the Laplacian matrix corresponding to the given graph.
    Skips all constant vector.
    Args:
        senders: The sender nodes of the graph
        receivers: The receiver nodes of the graph
        weights: The weights of the edges
        k: number of eigenvalue/vector pairs (excluding trivial eigenvector)
        n: # of nodes (optional)
    Returns:
        eigen_values: array of eigenvalues
        eigen_vectors: array of eigenvectors
    """
    m = senders.shape[0]
    if weights is None:
        weights = np.ones(m)

    if n is None:
        n = senders.max()
        if receivers.max() > n:
            n = receivers.max()
        n += 1

    lap_mat = laplacian_matrix(senders, receivers, weights, n=n)
    # n = lap_mat.shape[0]
    k = min(n - 2, k + 1)
    # rows of eigenv correspond to graph nodes, cols correspond to eigenvalues
    eigenvals, eigenvecs = sp.sparse.linalg.eigs(lap_mat, k=k, which="SM")
    eigenvals = np.real(eigenvals)
    eigenvecs = np.real(eigenvecs)

    # sort eigenvectors in ascending order of eigenvalues
    sorted_idx = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_idx]
    eigenvecs = eigenvecs[:, sorted_idx]

    constant_eigenvec_idx = 0

    for i in range(0, k):
        # normalize the i^th eigenvector
        eigenvecs[:, i] = eigenvecs[:, i] / np.sqrt((eigenvecs[:, i] ** 2).sum())
        if eigenvecs[:, i].var() <= 1e-7:
            constant_eigenvec_idx = i

    non_constant_idx = [*range(0, k)]
    non_constant_idx.remove(constant_eigenvec_idx)

    eigenvals = eigenvals[non_constant_idx]
    eigenvecs = eigenvecs[:, non_constant_idx]

    return eigenvals, eigenvecs


def add_expander_edges(g, degree, max_num_iters=100):
    nnodes = g.num_nodes()
    rng = np.random.default_rng()
    eig_val = -1
    eig_val_lower_bound = max(0, 2 * degree - 2 * math.sqrt(2 * degree - 1) - 0.1)
    max_eig_val_so_far = -1
    max_senders = []
    max_receivers = []
    cur_iter = 1
    if g.num_nodes() <= degree:
        degree = nnodes - 1
    if nnodes <= 10:
        for i in range(nnodes):
            for j in range(nnodes):
                if i != j:
                    max_senders.append(i)
                    max_receivers.append(j)
    else:
        while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
            senders = [*range(0, nnodes)] * degree
            receivers = rng.permutation(senders).tolist()
            senders, receivers = [*senders, *receivers], [*receivers, *senders]
            [eig_val, _] = laplacian_eigenv(
                np.array(senders), np.array(receivers), k=1, n=nnodes
            )
            if len(eig_val) == 0:
                eig_val = 0
            else:
                eig_val = eig_val[0]
            if eig_val > max_eig_val_so_far:
                max_eig_val_so_far = eig_val
                max_senders = senders
                max_receivers = receivers
            cur_iter += 1
    non_loops = [
        *filter(
            lambda i: max_senders[i] != max_receivers[i], range(0, len(max_senders))
        )
    ]
    senders = np.array(max_senders)[non_loops]
    receivers = np.array(max_receivers)[non_loops]
    max_senders = torch.tensor(max_senders, dtype=torch.long)
    max_receivers = torch.tensor(max_receivers, dtype=torch.long)

    # add to g
    g.add_edges(max_senders, max_receivers, etype="expander")
