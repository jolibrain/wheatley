import torch
from torch_geometric.data import Data, Batch
from utils.utils import put_back_one_hot_encoding_unbatched, put_back_one_hot_encoding_batched
from models.tokengt.utils import get_laplacian_pe_simple
import dgl
import time


class AgentObservation:
    def __init__(self, graphs, use_dgl, glist=False):
        self.use_dgl = use_dgl
        self.graphs = graphs
        self.glist = glist

    def get_batch_size(self):
        if self.use_dgl:
            if self.glist:
                return len(self.graphs)
            return self.graphs.batch_size
        return self.graph.num_graphs

    def get_n_nodes(self):
        if self.use_dgl:
            if self.glist:
                return int(sum([g.num_nodes() for g in self.graphs]) / len(self.graphs))
            else:
                return int(self.graphs.num_nodes() / self.graphs.batch_size)
        return int(self.graphs.num_nodes / self.graphs.num_graphs)

    @classmethod
    def build_graph(cls, n_edges, edges, nnodes, feats, bidir):
        edges0 = edges[0]
        edges1 = edges[1]
        type0 = [1] * n_edges
        if bidir:
            type1 = [2] * n_edges

            gnew = dgl.graph(
                (torch.cat([edges0, edges1]), torch.cat([edges1, edges0])),
                num_nodes=nnodes,
            )
        else:
            gnew = dgl.graph(
                (edges0, edges1),
                num_nodes=nnodes,
            )

        gnew.ndata["feat"] = feats
        if bidir:
            type0.extend(type1)
        gnew.edata["type"] = torch.LongTensor(type0)
        return gnew

    @classmethod
    def get_machine_id(cls, machine_one_hot):
        # print("machine_one_hot", machine_one_hot)

        return torch.max(machine_one_hot, dim=1)

    @classmethod
    def add_conflicts_cliques(cls, g, features, nnodes, max_n_machines):
        machineid = features[:, 5].long()
        m1 = machineid.unsqueeze(0).expand(nnodes, nnodes)
        # put m2 unaffected to -2 so that unaffected task are not considered in conflict
        m2 = torch.where(machineid == -1, -2, machineid).unsqueeze(1).expand(nnodes, nnodes)
        cond = torch.logical_and(torch.eq(m1, m2), torch.logical_not(torch.diag(torch.BoolTensor([True] * nnodes))))
        conflicts = torch.where(cond, 1, 0).nonzero(as_tuple=True)
        edgetype = machineid[conflicts[0]] + 5
        g.add_edges(conflicts[0], conflicts[1], data={"type": edgetype})
        return g

    @classmethod
    def from_gym_observation(
        cls,
        gym_observation,
        use_dgl=False,
        conflicts="att",
        max_n_machines=-1,
        add_self_loops=True,
        device=None,
        put_back_one_hot_encoding=False,
        do_batch=True,
        compute_laplacian_pe=False,
        laplacian_pe_cache=None,
        bidir=True,
    ):

        if not use_dgl:
            n_nodes = gym_observation["n_nodes"].long()
            n_edges = gym_observation["n_edges"].long()
            features = gym_observation["features"]

            # put back one_hot encoding
            features = put_back_one_hot_encoding_unbatched(features, max_n_machines)

            edge_index = gym_observation["edge_index"].long()
            # collating is much faster on cpu due to transfer of incs
            orig_device = features.device
            fcpu = features.to("cpu")
            eicpu = edge_index.to("cpu")
            graph = Batch.from_data_list(
                [Data(fcpu[i, : n_nodes[i], :], eicpu[i, : n_edges[i], :]) for i in range(fcpu.shape[0])]
            )
            return cls(graph.to(orig_device), use_dgl)
        else:
            # here again, batching on CPU...
            n_nodes = gym_observation["n_nodes"].to(torch.device("cpu")).long()
            n_edges = gym_observation["n_edges"].to(torch.device("cpu")).long()
            edge_index = gym_observation["edge_index"].to(torch.device("cpu")).long()
            orig_feat = gym_observation["features"].to(torch.device("cpu"))

            graphs = []
            if do_batch:
                batch_num_nodes = []
                batch_num_edges = []

            for i, nnodes in enumerate(n_nodes):
                features = orig_feat[i, :nnodes, :]
                gnew = cls.build_graph(
                    n_edges[i], edge_index[i, :, : n_edges[i].item()], nnodes.item(), orig_feat[i, : nnodes.item(), :], bidir
                )

                if conflicts == "clique":
                    gnew = AgentObservation.add_conflicts_cliques(gnew, features, nnodes.item(), max_n_machines)
                if add_self_loops:
                    gnew = dgl.add_self_loop(gnew, edge_feat_names=["type"], fill_data=0)
                if compute_laplacian_pe:
                    gnew.ndata["laplacian_pe"] = get_laplacian_pe_simple(gnew, laplacian_pe_cache)
                gnew = gnew.to(device)
                if put_back_one_hot_encoding:
                    gnew.ndata["feat"] = put_back_one_hot_encoding_batched(gnew.ndata["feat"], None, max_n_machines)
                graphs.append(gnew)
                if do_batch:
                    batch_num_nodes.append(gnew.num_nodes())
                    batch_num_edges.append(gnew.num_edges())

            if do_batch:
                graph = dgl.batch(graphs)
                graph.set_batch_num_nodes(torch.tensor(batch_num_nodes))
                graph.set_batch_num_edges(torch.tensor(batch_num_edges))

                return cls(graph, use_dgl, glist=False)
            else:
                return cls(graphs, use_dgl, glist=True)

    def to_graph(self):
        """
        Returns the batched graph associated with the observation.
        """
        return self.graphs
