#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#
import torch


def learned_graph_pool(
    g,
    node_offset,
    batch_size,
    origbnn,
    input_dim_features_extractor,
    typeid,
    revtypeid,
    inverse_pooling=False,
):
    poolnodes = list(range(node_offset, node_offset + batch_size))
    g.add_nodes(
        batch_size,
        data={"feat": torch.zeros((batch_size, input_dim_features_extractor))},
    )
    ei0 = []
    ei1 = []
    startnode = 0
    for i in range(batch_size):
        ei0 += [node_offset + i] * origbnn[i]
        ei1 += list(range(startnode, startnode + origbnn[i]))
        startnode += origbnn[i]

    g.add_edges(
        list(range(node_offset, node_offset + batch_size)),
        list(range(node_offset, node_offset + batch_size)),
        data={"type": torch.LongTensor([0] * batch_size)},
    )

    g.add_edges(ei1, ei0, data={"type": torch.LongTensor([typeid] * len(ei0))})
    if inverse_pooling:
        g.add_edges(ei0, ei1, data={"type": torch.LongTensor([revtypeid] * len(ei0))})

    return g, poolnodes, node_offset + batch_size


def node_conflicts(
    g,
    node_offset,
    batch_size,
    origbnn,
    first_res_index,
    max_n_resources,
    input_dim_features_extractor,
    edge_type_offset,
):
    resources_used = g.ndata["feat"][:, first_res_index:]
    num_resources = resources_used.shape[1]
    resource_nodes = list(range(node_offset, node_offset + num_resources * batch_size))
    batch_id = []
    for i, nn in enumerate(origbnn):
        batch_id.extend([i] * nn)
    batch_id = torch.IntTensor(batch_id)
    g.add_nodes(
        num_resources * batch_size,
        data={
            "feat": torch.zeros(
                (
                    batch_size * max_n_resources,
                    input_dim_features_extractor,
                )
            )
        },
    )
    idxaffected = torch.where(resources_used != 0)
    consumers = idxaffected[0]
    nconsumers = consumers.shape[0]
    resource_start_per_batch = []
    for i in range(batch_size):
        resource_start_per_batch.append(node_offset + num_resources * i)
    resource_start_per_batch = torch.IntTensor(resource_start_per_batch)
    resource_index = idxaffected[1] + resource_start_per_batch[batch_id[consumers]]

    rntype = torch.LongTensor(
        list(range(num_resources)) * batch_size
    ) + torch.LongTensor([edge_type_offset] * len(resource_nodes))

    g.add_edges(
        resource_nodes,
        resource_nodes,
        # we could use different self loop type per resource
        data={"type": rntype},
        # data={"type": torch.LongTensor([10] * len(resource_nodes))},
    )
    rc = torch.gather(resources_used[consumers], 1, idxaffected[1].unsqueeze(1)).expand(
        nconsumers, 2
    )

    g.add_edges(
        consumers,
        resource_index,
        data={
            # "type": torch.LongTensor([10] * nconsumers),
            "type": torch.LongTensor([edge_type_offset + num_resources] * nconsumers)
            + idxaffected[1].long(),
            "rid": idxaffected[1].int(),
            "att_rc": rc,
        },
    )
    g.add_edges(
        resource_index,
        consumers,
        data={
            # "type": torch.LongTensor([11] * nconsumers),
            "type": torch.LongTensor(
                [edge_type_offset + 2 * num_resources] * nconsumers
            )
            + idxaffected[1].long(),
            "rid": idxaffected[1].int(),
            "att_rc": rc,
        },
    )
    node_offset += num_resources * batch_size
    return g, resource_nodes, node_offset + num_resources * batch_size


def vnode(
    g, node_offset, batch_size, input_dim_features_extractor, origbnn, etype, revetype
):
    vnodes = list(range(node_offset, node_offset + batch_size))
    g.add_nodes(
        batch_size,
        data={"feat": torch.zeros((batch_size, input_dim_features_extractor))},
    )
    ei0 = []
    ei1 = []
    startnode = 0
    for i in range(batch_size):
        ei0 += [node_offset + i] * origbnn[i]
        ei1 += list(range(startnode, startnode + origbnn[i]))
        startnode += origbnn[i]

    # g.add_edges(
    #     list(range(node_offset, node_offset + batch_size)),
    #     list(range(node_offset, node_offset + batch_size)),
    #     data={"type": torch.LongTensor([0] * batch_size)},
    # )

    g.add_edges(ei1, ei0, data={"type": torch.LongTensor([etype] * len(ei0))})
    g.add_edges(ei0, ei1, data={"type": torch.LongTensor([revetype] * len(ei0))})

    return g, vnodes, node_offset + batch_size


def rewire(
    g,
    learned_graph_pooling,
    node_conflicting,
    vnoding,
    batch_size,
    input_dim_features_extractor,
    max_n_resources,
    poolnodeetype,
    poolnoderevetype,
    first_res_index,
    resource_edge_type_offset,
    vnodeetype,
    vnoderevetype,
):
    node_offset = g.num_nodes()
    origbnn = g.batch_num_nodes()

    if learned_graph_pooling:
        g, poolnodes, node_offset = learned_graph_pool(
            g,
            node_offset,
            batch_size,
            origbnn,
            input_dim_features_extractor,
            poolnodeetype,
            poolnoderevetype,
        )
    else:
        poolnodes = None

    if node_conflicting:
        g, resource_nodes, node_offset = node_conflicts(
            g,
            node_offset,
            batch_size,
            origbnn,
            first_res_index,
            max_n_resources,
            input_dim_features_extractor,
            resource_edge_type_offset,
        )
    else:
        resource_nodes = None

    if vnoding:
        g, vnodes, node_offset = vnode(
            g,
            node_offset,
            batch_size,
            input_dim_features_extractor,
            origbnn,
            vnodeetype,
            vnoderevetype,
        )
    else:
        vnodes = None

    return g, poolnodes, resource_nodes, vnodes
