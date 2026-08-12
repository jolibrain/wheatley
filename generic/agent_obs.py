import pickle
import torch

from torch_geometric.data import HeteroData, Data, Batch


class AgentObservation:
    def __init__(self, simg, rewire_params, custom_rewirer=None):
        self.learned_graph_pooling = rewire_params["graph_pooling"]
        self.do_bidir = rewire_params["bidir"]
        self.vnoding = rewire_params["vnoding"]
        self.self_loops = rewire_params["self_loops"]
        if custom_rewirer is not None:
            self.g = custom_rewirer.rewire(simg, rewire_params)
        else:
            self.g = simg
        self.generic_rewire()

        self.node_types = (
            rewire_params["node_types"] if "node_types" in rewire_params else "n"
        )
        # if lappe is not None:
        # subg = self.g._graph.edge_type_subgraph(
        #     [("n", "free", "n"), ("n", "rfree", "n"), ("n", "pool", "poolnode")]
        # )
        # addLapPe = AddLaplacianEigenvectorPE(lappe)
        # subgh = subg.to_homogeneous()
        # subgh = addLapPe(subgh)
        #     self.g._graph.laplacian_eigenvector_pe = subgh.laplacian_eigenvector_pe
        self.homo = self.g.to_homogeneous()

    def save(self, fname):
        data = {
            "g": self.g._graph.to_dict(),
            "homo": self.homo.to_dict(),
            "learned_graph_pooling": self.learned_graph_pooling,
            "do_bidir": self.do_bidir,
            "vnoding": self.vnoding,
            "self_loops": self.self_loops,
            "node_types": self.node_types,
        }
        torch.save(data, fname, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, fname, pyggraphtype):
        d = torch.load(fname, weights_only=False)
        o = cls.__new__(cls)
        # o.g = PYGGraph.__new__(PYGGraph)
        o.g = pyggraphtype.__new__(pyggraphtype)
        o.g._graph = HeteroData.from_dict(d["g"])
        o.g.cache = False
        o.homo = Data.from_dict(d["homo"])
        o.learned_graph_pooling = d["learned_graph_pooling"]
        o.do_bidir = d["do_bidir"]
        o.vnoding = d["vnoding"]
        o.self_loops = d["self_loops"]
        o.node_types = d["node_types"]
        return o

    def sloops(self):
        for ntype in self.g._graph.node_types:
            nnodes = self.g.num_nodes(ntype)
            if nnodes != 0:
                e0 = torch.arange(nnodes)
                e1 = torch.arange(nnodes)
                self.g.add_edges(
                    e0, e1, etype="self_" + ntype, node_type1=ntype, node_type2=ntype
                )

    def generic_rewire(self):
        if self.do_bidir:
            self.bidir()

        if self.self_loops:
            self.sloops()

        if self.learned_graph_pooling in ["learn", "learninv"]:
            inverse_pooling = self.learned_graph_pooling == "learninv"
            self.learned_graph_pool(inverse_pooling)

        if self.vnoding:
            self.vnode()

    def learned_graph_pool(
        self,
        inverse_pooling=False,
    ):
        self.g.add_nodes(1, node_type="poolnode")

        for nt in self.g._graph.node_types:
            if nt == "poolnode":
                continue
            if self.g.num_nodes(nt) != 0:
                ei0 = list(range(self.g.num_nodes(nt)))
                ei1 = [0] * self.g.num_nodes(nt)

                self.g.add_edges(
                    torch.tensor(ei0, dtype=torch.int64),
                    torch.tensor(ei1, dtype=torch.int64),
                    etype="pool",
                    node_type1=nt,
                    node_type2="poolnode",
                )
                if inverse_pooling:
                    self.g.add_edges(
                        torch.tensor(ei1),
                        torch.tensor(ei0),
                        etype="rpool",
                        node_type1="poolnode",
                        node_type2=nt,
                    )

        # self loops
        if self.self_loops:
            self.g.add_edges(
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                etype="selfpool",
                node_type1="poolnode",
                node_type2="poolnode",
            )

    def vnode(self):
        self.g.add_nodes(1, node_type="vnode")
        for nt in self.node_types:
            if self.g.num_nodes(nt) != 0:
                ei0 = list(range(self.g.num_nodes(nt)))
                ei1 = [0]

                self.g.add_edges(
                    ei0, ei1, etype="vnode", node_type1=nt, node_type2="vnode"
                )
                self.g.add_edges(
                    ei1, ei0, etype="rvnode", node_type1="vnode", node_type2=nt
                )
                self.g.add_edges(
                    torch.tensor([0], dtype=torch.int64),
                    torch.tensor([0], dtype=torch.int64),
                    etype="selfvnode",
                    node_type1="vnode",
                    node_type2="vnode",
                )

    def bidir(self):
        for fet in self.g._graph.edge_types:
            nt1 = fet[0]
            nt2 = fet[2]
            et = fet[1]
            e0, e1, _ = self.g.edges(et, node_type1=nt1, node_type2=nt2)
            self.g.add_edges(e1, e0, etype="r" + et, node_type1=nt2, node_type2=nt1)
            edata = self.g.edata(fet)
            for dataid, data in edata.items():
                if dataid != "edge_index":
                    self.g.set_edata(
                        "r" + et, dataid, data, node_type1=nt2, node_type2=nt1
                    )


class AgentObservationBatch:
    def __init__(
        self, pygbatch_homo, aos, node_types, edge_types, num_nodes, num_edges
    ):
        # self.graphs = pygbatch
        self.homo_graphs = pygbatch_homo
        self.aos = aos
        # self.n_graphs = self.graphs._graph.num_graphs
        self.n_graphs = pygbatch_homo.num_graphs
        # self.total_num_nodes = self.graphs._graph.num_nodes
        self.total_num_nodes = pygbatch_homo.num_nodes
        self.total_num_edges = pygbatch_homo.num_edges
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_types = node_types
        self.edge_types = edge_types
        self.node_types_map = {}
        self.edge_types_map = {}
        for n, nt in enumerate(self.node_types):
            self.node_types_map[nt] = n
        for n, et in enumerate(self.edge_types):
            self.edge_types_map[et] = n

    @classmethod
    def from_aos(cls, aos):
        # glist = []
        hlist = []
        num_nodes_list = []
        num_edges_list = []
        if type(aos) is AgentObservation:
            aos = [aos]
        for ao in aos:
            # glist.append(ao.g)
            hlist.append(ao.homo)
            nn = ao.g.num_nodes()
            if ao.learned_graph_pooling in ["learn", "learninv"]:
                nn -= 1
            if ao.vnoding:
                nn -= 1
            num_nodes_list.append(nn)
            num_edges_list.append(ao.g.num_edges())
        # pygbatch = PYGBatchGraph(glist)
        pygbatch_homo = Batch.from_data_list(hlist)
        return cls(
            pygbatch_homo,
            aos,
            aos[0].g._graph.node_types,
            aos[0].g._graph.edge_types,
            num_nodes_list,
            num_edges_list,
        )

    def homogeneous(self):
        # homo = self.graphs.to_homogeneous()
        homo = self.homo_graphs
        nodesid = {}
        edgesid = {}
        for nt in self.node_types:
            nodesid[nt] = torch.where(homo.node_type == self.node_types_map[nt])[0]
        for et in self.edge_types:
            edgesid[et] = torch.where(homo.edge_type == self.edge_types_map[et])[0]

        return (
            homo,
            self.aos,
            # self.graphs._graph.num_graphs,
            homo.num_graphs,
            self.total_num_nodes,
            self.total_num_edges,
            self.num_nodes,
            self.num_edges,
            nodesid,
            edgesid,
        )
