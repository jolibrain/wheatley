import torch


# def node_conflicts(
#     g,
#     node_offset,
#     batch_size,
#     input_dim_features_extractor,
# ):
#     resources_used = g.ndata("resources")
#     num_resources = resources_used.shape[1]
#     resource_nodes = torch.arange(node_offset, node_offset + num_resources * batch_size)
#     batch_id = g.batch_id()
#     data = torch.empty(
#         (
#             batch_size * num_resources,
#             input_dim_features_extractor,
#         )
#     )

#     # data[:, :] = torch.LongTensor(list(range(num_resources)) * batch_size).unsqueeze(1)
#     g.add_nodes(num_resources * batch_size, "feat", data)
#     bi = []
#     for i in range(batch_size):
#         bi.extend([i] * num_resources)
#     g.add_batchinfo(torch.tensor(bi))

#     idxaffected = torch.where(resources_used != 0)
#     consumers = idxaffected[0]
#     nconsumers = consumers.shape[0]
#     resource_start_per_batch = [
#         node_offset + num_resources * i for i in range(batch_size)
#     ]
#     resource_start_per_batch = torch.IntTensor(resource_start_per_batch)
#     resource_index = idxaffected[1] + resource_start_per_batch[batch_id[consumers]]

#     rc = torch.gather(resources_used[consumers], 1, idxaffected[1].unsqueeze(1)).expand(
#         nconsumers, 2
#     )

#     g.add_edges(
#         consumers,
#         resource_index,
#         etype="nodeconf",
#         data={
#             "rid": idxaffected[1].int(),
#             "att_rc": rc,
#         },
#     )
#     g.add_edges(
#         resource_index,
#         consumers,
#         etype="rnodeconf",
#         data={
#             "rid": idxaffected[1].int(),
#             "att_rc": rc,
#         },
#     )

#     # find unused resources
#     # ad local self loops
#     # unused_resources = resource_nodes[
#     #     torch.where(g.in_degrees(v=resource_nodes, etype="nodeconf") == 0)[0]
#     # ]
#     unused_resources = resource_nodes[
#         torch.where(torch.all(resources_used == 0, dim=0))
#     ]

#     g.add_edges(
#         unused_resources,
#         unused_resources,
#         etype="selfres",
#         # data={
#         # "type": torch.LongTensor(
#         #     [AgentObservation.edgeType["self"]] * unused_resources.shape[0]
#         # ),
#         # "rid": torch.zeros_like(unused_resources, dtype=torch.int),
#         # "att_rc": torch.zeros(unused_resources.shape[0], 2),
#         # },
#     )
#     return g, resource_nodes, node_offset + num_resources * batch_size


class PspRewirer:
    def __init__(self):
        pass

    def rewire(self, g, rewire_params):
        resources_used = g.ndata("resources")
        num_resources = resources_used.shape[1]
        g.add_nodes(num_resources, node_type="resource")
        g.set_ndata("resource_id", torch.arange(num_resources), node_type="resource")
        idxaffected = torch.where(resources_used != 0)
        consumers = idxaffected[0]
        # nconsumers = consumers.shape[0]
        resource_index = idxaffected[1]
        rc = torch.gather(
            resources_used[consumers], 1, idxaffected[1].unsqueeze(1)
        ).squeeze(-1)  # .expand(nconsumers, 2)
        g.add_edges(
            consumers,
            resource_index,
            etype="uses",
            data={
                "rid": idxaffected[1].int(),
                "att_uses": rc,
            },
            node_type1="n",
            node_type2="resource",
        )
        # g.add_edges(
        #     resource_index,
        #     consumers,
        #     etype="rnodeconf",
        #     data={
        #         "rid": idxaffected[1].int(),
        #         "att_rc": rc,
        #     },
        #     node_type1="resnode",
        #     node_type2="n",
        # )
        # resource_nodes = torch.arange(num_resources)
        # unused_resources = resource_nodes[
        #     torch.where(torch.all(resources_used == 0, dim=0))
        # ]

        # g.add_edges(
        #     unused_resources,
        #     unused_resources,
        #     etype="selfres",
        #     node_type1="resnode",
        #     node_type2="resnode",
        # )

        return g
