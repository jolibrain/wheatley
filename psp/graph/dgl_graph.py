from .graph import Graph

import torch
import copy

from dgl import (
    heterograph,
    save_graphs,
    load_graphs,
    node_subgraph,
    NID,
    to_homogeneous,
    batch,
)


class DGLGraph(Graph):
    def __init__(
        self,
        problem_edges,
        num_nodes,
        factored_rp,
        observe_conflicts_as_cliques,
        device,
    ):
        self.pred_cache = []
        self.suc_cache = []
        self.indeg_cache = []
        self._global_data = {}
        pe = torch.tensor(problem_edges, dtype=torch.int64).t()
        gd = {
            ("n", "prec", "n"): (pe[0], pe[1]),
            ("n", "rprec", "n"): ((), ()),
            ("n", "rc", "n"): ((), ()),
            ("n", "rp", "n"): ((), ()),
            ("n", "rrp", "n"): ((), ()),
            ("n", "pool", "n"): ((), ()),
            ("n", "rpool", "n"): ((), ()),
            ("n", "self", "n"): ((), ()),
            ("n", "vnode", "n"): ((), ()),
            ("n", "rvnode", "n"): ((), ()),
            ("n", "nodeconf", "n"): ((), ()),
            ("n", "rnodeconf", "n"): ((), ()),
            # ("global_data", "null", "global_data"): ((), ()),
        }
        self._graph = heterograph(
            gd,
            device=device,
            # num_nodes_dict={"n": num_nodes, "global_data": 0},
            num_nodes_dict={"n": num_nodes},
        )

        # START workaround unset attributes schemes
        if factored_rp:
            self._graph.add_edges(
                0,
                0,
                etype="rp",
                data={"r": torch.zeros(self.env_specification.max_n_resources * 3)},
            )
        else:
            self._graph.add_edges(
                0,
                0,
                etype="rp",
                data={
                    "r": torch.zeros(
                        (1, 4),
                        dtype=torch.float,
                    )
                },
            )
        eid = self._graph.edge_ids(0, 0, etype="rp")
        self._graph.remove_edges(eid, etype="rp")

        if observe_conflicts_as_cliques:
            self._graph.add_edges(
                0,
                0,
                etype="rc",
                data={
                    "rid": torch.tensor([0]),
                    "val": torch.tensor([0.5]),
                    "valr": torch.tensor([0.3]),
                },
            )
            eid = self._graph.edge_ids(0, 0, etype="rc")
            self._graph.remove_edges(eid, etype="rc")
        # END of woraround unset attribute schemes

        for n in range(self._graph.num_nodes(ntype="n")):
            # self.pred_cache[n] = self.graph.predecessors(n, etype="prec")
            self.pred_cache.append(
                self._graph._graph.predecessors(self._graph.get_etype_id("prec"), n)
            )
            # self.suc_cache[n] = self.graph.successors(n, etype="prec")
            self.suc_cache.append(
                self._graph._graph.successors(self._graph.get_etype_id("prec"), n)
            )
            self.indeg_cache.append(self._graph.in_degrees(n, etype="prec"))

        self._graph.ndata["job"] = torch.zeros(
            num_nodes, dtype=torch.int, device=device
        )

        self._graph.ndata["durations"] = torch.zeros(
            (num_nodes, 3), dtype=torch.float, device=device
        )

    def ndata(self, featname=None):
        if featname is None:
            return self._graph.ndata
        return self._graph.ndata[featname]

    def set_ndata(self, featname, t):
        self._graph.ndata[featname] = t

    def num_nodes(self):
        return self._graph.num_nodes(ntype="n")

    def predecessors(self, nid):
        return self.pred_cache[nid]

    def successors(self, nid):
        return self.suc_cache[nid]

    def indeg(self, nid):
        return self.indeg_cache[nid]

    def in_degrees(self):
        return self._graph.in_degrees(etype="prec")

    def out_degrees(self):
        return self._graph.out_degrees(etype="prec")

    def set_global_data(self, featname, data):
        # self._graph.add_nodes(1, ntype="global_data")
        # self._graph.ndata[featname] = {
        #     "global_data": data.clone().detach().unsqueeze(0)
        # }
        self._global_data[featname] = data.clone().detach()

    def global_data(self, featname=None):
        if featname is None:
            return self._global_data
        return self._global_data[featname]

    def edges(self, etype):
        return self._graph.edges(etype=etype, form="all")

    def remove_edges(self, eid, etype):
        self._graph.remove_edges(eid, etype=etype)

    def add_edges(self, sources, destinations, etype, data=None):
        self._graph.add_edges(sources, destinations, data=data, etype=etype)

    def num_edges(self, etype):
        return self._graph.num_edges(etype=etype)

    def edata(self, etype, dataid):
        if etype is None:
            return self._graph.edata[dataid]
        return self._graph.edata[dataid][("n", etype, "n")]

    def set_edata(self, etype, featname, data, index=None):
        if index is not None:
            self._graph.edges[etype].data[featname][:, index] = data
        else:
            self._graph.edges[etype].data[featname] = data

    @classmethod
    def batch(cls, graphlist, num_nodes, num_edges):
        return DGLBatchGraph(graphlist, num_nodes, num_edges)

    def save(self, fname):
        save_graphs(fname + "_graph", [self._graph])
        torch.save(self._global_data, fname + "_global_data")

    @classmethod
    def load(cls, fname):
        ng = cls.__new__(cls)
        ng._graph = load_graphs(fname + "_graph")[0][0]
        ng._global_data = torch.load(fname + "_global_data", weights_only=True)
        return ng

    def fill_void(self):
        pass

    def node_subgraph(self, nodes_to_keep):
        return node_subgraph(self, nodes_to_keep)

    def subid_to_origid(self, i):
        return self._graph.ndata[NID][i].item()

    def fullmask_to_submask(self, mask):
        return mask[self._graph.ndata[NID]]

    def clone(self):
        return copy.deepcopy(self)


class DGLBatchGraph(DGLGraph):
    def __init__(self, graphs, batch_num_nodes, batch_num_edges):
        self._graph = batch([g._graph for g in graphs])
        self._graph.set_batch_num_nodes(torch.tensor(batch_num_nodes))
        self._graph.set_batch_num_edges(batch_num_edges)
        self._batch_num_nodes = torch.tensor(batch_num_nodes)
        self._batch_num_edges = batch_num_edges

    def num_nodes(self):
        return self._graph.num_nodes()

    def batch_num_nodes(self):
        return self._batch_num_nodes

    def add_nodes(self, num_nodes, featname, data):
        self._graph.add_nodes(num_nodes, data={featname: data})

    def add_edges(self, sources, dests, etype, data=None):
        self._graph.add_edges(sources, dests, etype=etype, data=data)

    def in_degrees(self, v, etype):
        return self._graph.in_degrees(v=v, etype=etype)

    def add_self_loops(self):
        self._graph = self._graph.add_self_loop(etype="self")

    def to_homogeneous(self, ndata, edata):
        self._graph = to_homogeneous(
            self._graph, ndata=ndata, edata=edata, store_type=False
        )

    def to(self, device):
        self._graph = self._graph.to(device)
        return self
