#    MIT License
#
#    Copyright (c) Microsoft Corporation.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE
#
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import dgl.backend as F
import dgl
from packaging import version

# import time

# from numba import jit
# from tinydb import Query


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


# @jit(forceobj=True)
def get_laplacian_pe_simple(g, cache=None, k=50):

    if cache is not None:
        edges = g.edges()
        key = (tuple(edges[0].tolist()), tuple(edges[1].tolist()))
        if key in cache:
            return cache[key]

    # classical computation
    # start = time.time()
    if version.parse(dgl.__version__) >= version.parse("1.1.0"):
        A = g.adj_external(scipy_fmt="csr")  # adjacency matrix
    else:
        A = g.adj(scipy_fmt="csr")  # adjacency matrix
    N = sparse.diags(F.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)  # D^-1/2
    L = sparse.eye(g.num_nodes()) - N @ A @ N
    EigVal, EigVec = np.linalg.eigh(L.toarray())
    kpartition_indices = np.argpartition(EigVal, k + 1)[: k + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = np.real(EigVec[:, topk_indices])
    ret = torch.from_numpy(topk_EigVec).float()
    # print("classicaltime", time.time() - start)

    # sparse + iterative + approximated
    # start = time.time()
    # g2 = g
    # A = g2.adj(scipy_fmt="csr")  # adjacency matrix
    # N = sparse.diags(F.asnumpy(g2.in_degrees()).clip(1) ** -0.5, dtype=float)  # D^-1/2
    # L = sparse.eye(g2.num_nodes()) - N @ A @ N
    # EigVal2, EigVec2 = scipy.sparse.linalg.eigs(L, k=k + 1, sigma=0, tol=1e-8)
    # topk_indices2 = np.arange(k + 1)[EigVal2.argsort()][1:]
    # topk_eigvec2 = np.real(EigVec2[:, topk_indices2])
    # ret = torch.from_numpy(topk_eigvec2).float()
    # print("sparse time", time.time() - start)
    # exit()

    # select eigenvector

    # ret = torch.from_numpy(EigVec).float()
    if cache is not None:
        cachesize = len(cache) * ret.nelement() * ret.element_size()
        if cachesize < 5e9:
            cache[key] = ret
        else:
            # print("emptying laplacian pe cache", len(cache))
            cache = {}
            cache[key] = ret
    return ret


# def get_laplacian_pe(unbatched_graphs, max_n):
#     lap_eigvec = [get_laplacian_pe_simple(g) for g in unbatched_graphs]
#     lap_eigvec = torch.cat([torch.nn.functional.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvec])

#     return lap_eigvec
