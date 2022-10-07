import torch
from torch_geometric.data import Data, Batch
from utils.utils import put_back_one_hot_encoding_unbatched
import dgl
import time


class AgentObservation:
    def __init__(self, graph, use_dgl):
        self.use_dgl = use_dgl
        self.graph = graph

    def get_batch_size(self):
        if self.use_dgl:
            return self.graph.batch_size
        return self.graph.num_graphs

    def get_n_nodes(self):
        if self.use_dgl:
            return int(self.graph.num_nodes() / self.graph.batch_size)
        return int(self.graph.num_nodes / self.graph.num_graphs)

    @classmethod
    def build_graph(cls, n_edges, edges, nnodes, feats):
        edges0 = edges[0]
        edges1 = edges[1]
        type0 = [1] * n_edges
        type1 = [2] * n_edges

        gnew = dgl.graph(
            (torch.cat([edges0, edges1]), torch.cat([edges1, edges0])),
            num_nodes=nnodes,
        )
        gnew.ndata["feat"] = feats
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
    def from_gym_observation(cls, gym_observation, use_dgl=False, conflicts="att", max_n_machines=-1):

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
            orig_device = gym_observation["features"].device
            n_nodes = gym_observation["n_nodes"].to(torch.device("cpu")).long()
            n_edges = gym_observation["n_edges"].to(torch.device("cpu")).long()
            edge_index = gym_observation["edge_index"].to(torch.device("cpu")).long()
            orig_feat = gym_observation["features"].to(torch.device("cpu"))

            graphs = []
            batch_num_nodes = []
            batch_num_edges = []
            startnode = 0
            for i, nnodes in enumerate(n_nodes):
                features = orig_feat[i, :nnodes, :]
                gnew = cls.build_graph(
                    n_edges[i], edge_index[i, :, : n_edges[i].item()], nnodes.item(), orig_feat[i, : nnodes.item(), :]
                )

                if conflicts == "clique":
                    gnew = AgentObservation.add_conflicts_cliques(gnew, features, nnodes.item(), max_n_machines)

                graphs.append(gnew)
                batch_num_nodes.append(gnew.num_nodes())
                batch_num_edges.append(gnew.num_edges())
            graph = dgl.batch(graphs)

            graph = dgl.add_self_loop(graph, edge_feat_names=["type"], fill_data=0)

            graph.set_batch_num_nodes(torch.tensor(batch_num_nodes))
            graph.set_batch_num_edges(torch.tensor(batch_num_edges))
            # graph.pin_memory_()
            # graph = graph.to(orig_device)

            return cls(graph, use_dgl)

    def to_graph(self):
        """
        Returns the batched graph associated with the observation.
        """
        return self.graph
