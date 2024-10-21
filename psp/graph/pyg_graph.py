from .graph import Graph

import torch
import torch_geometric
from torch_geometric.data import HeteroData, Data, Batch
import pickle
import io


class PYGGraph(Graph):
    def __init__(
        self,
        problem_edges,
        num_nodes,
        factored_rp,
        observe_conflicts_as_cliques,
        device,
    ):
        self._graph = HeteroData()
        self._graph["n"].num_nodes = num_nodes
        self._graph["n", "prec", "n"].edge_index = torch.tensor(
            problem_edges, dtype=torch.int64
        ).t()

        self._graph.to(device)
        self.factored_rp = factored_rp
        self.observe_conflicts_as_cliques = observe_conflicts_as_cliques

        self.compute_caches()

        self._graph["n"]["job"] = torch.zeros(num_nodes, dtype=torch.int, device=device)
        self._graph["n"]["durations"] = torch.zeros(
            (num_nodes, 3), dtype=torch.float, device=device
        )
        self.cache = False

    def compute_caches(self):
        self._pred_cache = []
        self._suc_cache = []
        self._indeg_cache = []
        self._outdeg_cache = []
        for n in range(self._graph["n"].num_nodes):
            preds = self._graph["n", "prec", "n"].edge_index[0][
                torch.where(self._graph["n", "prec", "n"].edge_index[1] == n)[0]
            ]
            self._pred_cache.append(preds)
            self._indeg_cache.append(len(preds))
            sucs = self._graph["n", "prec", "n"].edge_index[1][
                torch.where(self._graph["n", "prec", "n"].edge_index[0] == n)[0]
            ]
            self._suc_cache.append(sucs)
            self._outdeg_cache.append(len(sucs))
        self._indeg_cache = torch.tensor(self._indeg_cache)
        self._outdeg_cache = torch.tensor(self._outdeg_cache)
        self.cache = True

    def ndata(self, featname=None):
        if isinstance(self._graph, Data):
            if featname is None:
                return self._graph
            return self._graph[featname]
        if featname is None:
            return self._graph["n"]
        return self._graph["n"][featname]

    def set_ndata(self, featname, t):
        self._graph["n"][featname] = t

    def set_edata(self, etype, featname, data, index=None):
        if index is None:
            self._graph["n", etype, "n"][featname] = data
        else:
            self._graph["n", etype, "n"][featname][:, index] = data

    def num_nodes(self):
        if self._graph["n"].num_nodes is None:
            return 0
        return self._graph["n"].num_nodes

    def predecessors(self, nid):
        if not self.cache:
            self.compute_caches()
        return self._pred_cache[nid]

    def successors(self, nid):
        if not self.cache:
            self.compute_caches()
        return self._suc_cache[nid]

    def indeg(self, nid):
        if not self.cache:
            self.compute_caches()
        return self._indeg_cache[nid]

    def in_degrees(self, v=None, etype=None):
        if (etype == "prec" or etype is None) and v is None:
            return self._indeg_cache
        indegs = []
        for n in v:
            preds = self._graph["n", etype, "n"].edge_index[0][
                torch.where(self._graph["n", etype, "n"].edge_index[1] == n)[0]
            ]
            indegs.append(len(preds))
        return torch.tensor(indegs, dtype=torch.int64)

    def out_degrees(self):
        if not self.cache:
            self.compute_caches()
        return self._outdeg_cache

    def set_global_data(self, featname, data):
        self._graph[featname] = data

    def global_data(self, featname=None):
        if featname is None:
            return self._graph
        return self._graph[featname]

    def edges(self, etype):
        if self._graph["n", etype, "n"].num_edges == 0:
            # return torch.empty(0, 3, dtype=torch.int64)
            return (
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
            )
        else:
            edges = self._graph["n", etype, "n"].edge_index
            # return torch.cat(
            #     [edges, torch.tensor(list(range(edges.shape[1]))).unsqueeze(0)],
            #     dim=0,
            # )
            return (
                edges[0],
                edges[1],
                torch.tensor(list(range(edges.shape[1])), dtype=torch.int64),
            )

    def remove_edges(self, eid, etype):
        if len(eid) == 0:
            return
        # mask = torch.tensor([True] * self._graph["n", etype, "n"].edge_index.shape[1])
        # mask[eid] = False
        mask = torch_geometric.utils.index_to_mask(
            eid, size=self._graph["n", etype, "n"].num_edges
        )
        self._graph["n", etype, "n"].edge_index = (
            #            self._graph["n", etype, "n"].edge_index[:, mask].clone()
            torch_geometric.utils.mask_select(
                self._graph["n", etype, "n"].edge_index, 1, torch.logical_not(mask)
            )
        )

    def add_nodes(self, num_nodes, featname, data):
        self._graph["n"][featname] = torch.cat([self._graph["n"][featname], data])

    def add_batchinfo(self, batchinfo):
        self._graph["n"]["batch"] = torch.cat([self._graph["n"]["batch"], batchinfo])

    def add_edges(self, sources, destinations, etype, data=None):
        new_edges = torch.stack([sources, destinations], dim=0)
        if self._graph["n", etype, "n"].num_edges == 0:
            self._graph["n", etype, "n"].edge_index = new_edges
            if data is not None:
                for key, value in data.items():
                    self._graph["n", etype, "n"][key] = data[key]

        else:
            self._graph["n", etype, "n"].edge_index = torch.cat(
                [self._graph["n", etype, "n"].edge_index, new_edges], dim=1
            )
            if data is not None:
                for key, value in data.items():
                    self._graph["n", etype, "n"][key] = torch.cat(
                        [self._graph["n", etype, "n"][key], data[key]], dim=0
                    )

    def num_edges(self, etype):
        return self._graph["n", etype, "n"].num_edges

    def edata(self, etype, dataid):
        if isinstance(self._graph, Data):
            if etype is None:
                return self._graph[dataid]
            return self._graph[etype][dataid]
        if etype is None:
            return self._graph["n"][dataid]
        return self._graph["n", etype, "n"][dataid]

    @classmethod
    def batch(cls, graphlist, num_nodes, num_edges):
        return PYGBatchGraph(graphlist)

    def clone(self):
        g = PYGGraph.__new__(PYGGraph)
        g._graph = self._graph.clone()
        g.factored_rp = self.factored_rp
        g.observe_conflicts_as_cliques = self.observe_conflicts_as_cliques
        g.cache = False
        # g._pred_cache = [t.clone() for t in self._pred_cache]
        # g._suc_cache = [t.clone() for t in self._suc_cache]
        # g._indeg_cache = self._indeg_cache.clone()
        # g._outdeg_cache = self._outdeg_cache.clone()
        return g

    def save(self, fname):
        data = {
            "graph": self._graph.to_dict(),
            # "pred_cache": [
            #     self._pred_cache[i].tolist() for i in range(self.num_nodes())
            # ],
            # "suc_cache": [self._suc_cache[i].tolist() for i in range(self.num_nodes())],
            # "indeg_cache": self._indeg_cache.tolist(),
            # "outdeg_cache": self._outdeg_cache.tolist(),
            "factored_rp": self.factored_rp,
            "obs_cliques": self.observe_conflicts_as_cliques,
        }
        torch.save(data, fname, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def serialize(self):
        buf = io.BytesIO()
        data = {
            "graph": self._graph.to_dict(),
            # "pred_cache": [
            #     self._pred_cache[i].tolist() for i in range(self.num_nodes())
            # ],
            # "suc_cache": [self._suc_cache[i].tolist() for i in range(self.num_nodes())],
            # "indeg_cache": self._indeg_cache.tolist(),
            # "outdeg_cache": self._outdeg_cache.tolist(),
            "factored_rp": self.factored_rp,
            "obs_cliques": self.observe_conflicts_as_cliques,
        }
        torch.save(data, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        out = buf.getvalue()
        return out, len(out)

    @classmethod
    def load(cls, fname):
        d = torch.load(fname, weights_only=False)
        g = cls.__new__(cls)
        g._graph = HeteroData.from_dict(d["graph"])
        # g._pred_cache = [torch.tensor(d["pred_cache"][i]) for i in range(g.num_nodes())]
        # g._suc_cache = [torch.tensor(d["suc_cache"][i]) for i in range(g.num_nodes())]
        # g._indeg_cache = torch.tensor(d["indeg_cache"])
        # g._outdeg_cache = torch.tensor(d["outdeg_cache"])
        g.factored_rp = d["factored_rp"]
        g.observe_conflicts_as_cliques = d["obs_cliques"]
        g.cache = False
        return g

    @classmethod
    def deserialize(cls, bytearr):
        d = torch.load(io.BytesIO(bytearr), weights_only=False)
        g = cls.__new__(cls)
        g._graph = HeteroData.from_dict(d["graph"])
        # g._pred_cache = [torch.tensor(d["pred_cache"][i]) for i in range(g.num_nodes())]
        # g._suc_cache = [torch.tensor(d["suc_cache"][i]) for i in range(g.num_nodes())]
        # g._indeg_cache = torch.tensor(d["indeg_cache"])
        # g._outdeg_cache = torch.tensor(d["outdeg_cache"])
        g.factored_rp = d["factored_rp"]
        g.observe_conflicts_as_cliques = d["obs_cliques"]
        g.cache = False
        return g

    def fill_void(self):
        if not "edge_index" in self._graph["n", "prec", "n"]:
            self._graph["n", "prec", "n"].edge_index = torch.tensor(
                [[], []], dtype=torch.int64
            )
        if not "edge_index" in self._graph["n", "rprec", "n"]:
            self._graph["n", "rprec", "n"].edge_index = torch.tensor(
                [[], []], dtype=torch.int64
            )
        if not "edge_index" in self._graph["n", "rp", "n"]:
            self._graph["n", "rp", "n"].edge_index = torch.tensor(
                [[], []], dtype=torch.int64
            )
            self._graph["n", "rp", "n"].r = torch.empty(0, 4, dtype=torch.float)
        if not "edge_index" in self._graph["n", "rrp", "n"]:
            self._graph["n", "rrp", "n"].edge_index = torch.tensor(
                [[], []], dtype=torch.int64
            )
            self._graph["n", "rrp", "n"].r = torch.empty(0, 4, dtype=torch.float)

    def node_subgraph(self, nodes_to_keep):
        g = type(self).__new__(type(self))
        g._graph = self._graph.subgraph({"n": nodes_to_keep})
        g.factored_rp = self.factored_rp
        g.observe_conflicts_as_cliques = self.observe_conflicts_as_cliques
        g._kept_nodes = nodes_to_keep
        g.compute_caches()
        return g

    def subid_to_origid(self, action):
        return self._kept_nodes[action]

    def fullmask_to_submask(self, mask):
        return mask[self._kept_nodes]


class PYGBatchGraph(PYGGraph):
    def __init__(self, graphs):
        self._graph = Batch.from_data_list([g._graph for g in graphs])

    def batch_num_nodes(self):
        ptrs = self._graph["n"].ptr
        return ptrs[1:] - ptrs[:-1]
        # return self._graph._slice_dict["n"]["feat"][1:]

    def add_self_loops(self):
        l = list(range(self._graph.num_nodes))
        self._graph["n", "self", "n"].edge_index = torch.stack(
            [torch.tensor(l, dtype=torch.int64), torch.tensor(l, dtype=torch.int64)]
        )

    def to_homogeneous(self, ndata, edata):
        self._graph = self._graph.to_homogeneous(
            ndata, edata, add_node_type=False, add_edge_type=False, dummy_values=False
        )

    def to(self, device):
        self._graph = self._graph.to(device)
        return self
