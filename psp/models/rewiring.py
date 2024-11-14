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
from .agent_observation import AgentObservation


def learned_graph_pool(
    g,
    node_offset,
    resource_nodes,
    batch_size,
    origbnn,
    input_dim_features_extractor,
    typeid,
    revtypeid,
    inverse_pooling=False,
    graphobs=False,
):
    poolnodes = list(range(node_offset, node_offset + batch_size))
    data = torch.zeros((batch_size, input_dim_features_extractor))
    g.add_nodes(batch_size, "feat", data)
    g.add_batchinfo(torch.tensor([i for i in range(batch_size)]))
    ei0 = []
    ei1 = []
    startnode = 0
    for i in range(batch_size):
        ei0 += [node_offset + i] * origbnn[i]
        ei1 += list(range(startnode, startnode + origbnn[i]))
        startnode += origbnn[i]

    if resource_nodes is not None:
        num_res = len(resource_nodes) // batch_size
        for i in range(batch_size):
            ei1 += resource_nodes[i * num_res : (i + 1) * num_res]
            ei0 += [node_offset + i] * num_res

    if graphobs:
        g.add_edges(
            torch.tensor(ei1, dtype=torch.int64),
            torch.tensor(ei0, dtype=torch.int64),
            etype="pool",
        )
    else:
        g.add_edges(ei1, ei0, data={"type": torch.LongTensor([typeid] * len(ei0))})
    if inverse_pooling:
        if graphobs:
            g.add_edges(torch.tensor(ei0), torch.tensor(ei1), etype="rpool")
        else:
            g.add_edges(
                ei0, ei1, data={"type": torch.LongTensor([revtypeid] * len(ei0))}
            )

    # self loops
    g.add_edges(
        torch.tensor(
            list(range(node_offset, node_offset + batch_size)), dtype=torch.int64
        ),
        torch.tensor(
            list(range(node_offset, node_offset + batch_size)), dtype=torch.int64
        ),
        etype="self",
    )

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
    graphobs,
):
    if graphobs:
        resources_used = g.ndata("resources")
    else:
        resources_used = g.ndata["feat"][:, first_res_index:]
    num_resources = resources_used.shape[1]
    resource_nodes = list(range(node_offset, node_offset + num_resources * batch_size))
    batch_id = []
    for i, nn in enumerate(origbnn):
        batch_id.extend([i] * nn)
    batch_id = torch.IntTensor(batch_id)
    data = torch.zeros(
        (
            batch_size * num_resources,
            input_dim_features_extractor,
        )
    )
    data[:, :] = torch.LongTensor(list(range(num_resources)) * batch_size).unsqueeze(1)
    g.add_nodes(num_resources * batch_size, "feat", data)
    bi = []
    for i in range(batch_size):
        bi.extend([i] * num_resources)
    g.add_batchinfo(torch.tensor(bi))

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

    rc = torch.gather(resources_used[consumers], 1, idxaffected[1].unsqueeze(1)).expand(
        nconsumers, 2
    )

    if graphobs:
        g.add_edges(
            consumers,
            resource_index,
            etype="nodeconf",
            data={
                # "type": torch.LongTensor([edge_type_offset] * nconsumers),
                # "type": torch.LongTensor(
                #     [edge_type_offset + num_resources] * nconsumers
                # )
                # + idxaffected[1].long(),
                "rid": idxaffected[1].int(),
                "att_rc": rc,
            },
        )
        g.add_edges(
            resource_index,
            consumers,
            etype="rnodeconf",
            data={
                # "type": torch.LongTensor([edge_type_offset + 1] * nconsumers),
                # edge type per resource : not better, nres dependent : DISCARD
                # "type": torch.LongTensor(
                #     [edge_type_offset + 2 * num_resources] * nconsumers
                # )
                # + idxaffected[1].long(),
                "rid": idxaffected[1].int(),
                "att_rc": rc,
            },
        )

    else:
        g.add_edges(
            consumers,
            resource_index,
            data={
                "type": torch.LongTensor([edge_type_offset] * nconsumers),
                # edge type per resource : not better, nres dependent : DISCARD
                # "type": torch.LongTensor([edge_type_offset] * nconsumers)
                # + idxaffected[1].long(),
                "rid": idxaffected[1].int(),
                "att_rc": rc,
            },
        )
        g.add_edges(
            resource_index,
            consumers,
            data={
                "type": torch.LongTensor([edge_type_offset + 1] * nconsumers),
                # "type": torch.LongTensor(
                #     [edge_type_offset + num_resources] * nconsumers
                # )
                # + idxaffected[1].long(),
                "rid": idxaffected[1].int(),
                "att_rc": rc,
            },
        )

    # find unused resources
    # ad local self loops
    if graphobs:
        unused_resources = torch.tensor(resource_nodes)[
            torch.where(g.in_degrees(v=resource_nodes, etype="nodeconf") == 0)[0]
        ]

        g.add_edges(
            unused_resources,
            unused_resources,
            etype="self",
            # data={
            # "type": torch.LongTensor(
            #     [AgentObservation.edgeType["self"]] * unused_resources.shape[0]
            # ),
            # "rid": torch.zeros_like(unused_resources, dtype=torch.int),
            # "att_rc": torch.zeros(unused_resources.shape[0], 2),
            # },
        )
    else:
        unused_resources = torch.tensor(resource_nodes)[
            torch.where(g.in_degrees(v=resource_nodes) == 0)[0]
        ]
        g.add_edges(
            unused_resources,
            unused_resources,
            data={
                "type": torch.LongTensor(
                    [AgentObservation.edgeType["self"]] * unused_resources.shape[0]
                ),
                "rid": torch.zeros_like(unused_resources, dtype=torch.int),
                "att_rc": torch.zeros(unused_resources.shape[0], 2),
            },
        )
    return g, resource_nodes, node_offset + num_resources * batch_size


def vnode(
    g,
    node_offset,
    batch_size,
    input_dim_features_extractor,
    origbnn,
    etype,
    revetype,
    graphobs,
):
    vnodes = list(range(node_offset, node_offset + batch_size))
    data = torch.zeros((batch_size, input_dim_features_extractor))
    data[:, :4] = 3
    g.add_nodes(
        batch_size,
        "feat",
        data,
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

    if graphobs:
        g.add_edges(ei1, ei0, etype="vnode")
        g.add_edges(ei0, ei1, etype="rvnode")
    else:
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
    graphobs,
):
    node_offset = g.num_nodes()
    origbnn = g.batch_num_nodes()

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
            graphobs,
        )
    else:
        resource_nodes = None

    if learned_graph_pooling in ["learn", "learninv"]:
        inverse_pooling = learned_graph_pooling == "learninv"
        g, poolnodes, node_offset = learned_graph_pool(
            g,
            node_offset,
            resource_nodes,
            batch_size,
            origbnn,
            input_dim_features_extractor,
            poolnodeetype,
            poolnoderevetype,
            inverse_pooling=inverse_pooling,
            graphobs=graphobs,
        )
    else:
        poolnodes = None

    if vnoding:
        g, vnodes, node_offset = vnode(
            g,
            node_offset,
            batch_size,
            input_dim_features_extractor,
            origbnn,
            vnodeetype,
            vnoderevetype,
            graphobs,
        )
    else:
        vnodes = None

    return g, poolnodes, resource_nodes, vnodes


def homogeneous_edges(g, edgetypes, factored_rp, conflict_type, max_n_resources):
    # TODO create homogeneous features for all edges of interest
    # edge type :
    #    prec
    #   rprec
    #    rc
    #   rrc [same type as rc]
    #    rp
    #   rrp
    #    pool
    #   rpool
    #  nodeconf
    #  rnodeconf
    #    self
    # vnode
    # rvnode
    #
    # actual features:
    # type : int
    # rid : int
    # att_rp    : factorred: 3*max_res    nonfact: 3
    # att_rc   : 2
    for t in edgetypes.keys():
        g.set_edata(
            t,
            "type",
            torch.zeros((g.num_edges(etype=t)), dtype=torch.long) + edgetypes[t],
        )

        if t not in ["rc", "nodeconf", "rnodeconf"] or g.num_edges(etype=t) == 0:
            g.set_edata(t, "rid", torch.zeros((g.num_edges(etype=t)), dtype=torch.int))

        if (conflict_type != "node") or (t not in ["nodeconf", "rnodeconf"]):
            g.set_edata(
                t, "att_rc", torch.zeros((g.num_edges(etype=t), 2), dtype=torch.float)
            )

        if factored_rp:
            rpdata = torch.zeros(
                (g.num_edges(etype=t), 3 * max_n_resources), dtype=torch.float
            )
        else:
            rpdata = torch.zeros((g.num_edges(etype=t), 3), dtype=torch.float)
        g.set_edata(t, "att_rp", rpdata)

    if g.num_edges(etype="rp") != 0:
        if factored_rp:
            g.set_edata("rp", "att_rp", g.edata("rp", "r"))
            g.set_edata("rrp", "att_rp", g.edata("rrp", "r"))
        else:
            g.set_edata("rp", "rid", (g.edata("rp", "r")[:, 0] + 1).int())
            g.set_edata("rrp", "rid", (g.edata("rrp", "r")[:, 0] + 1).int())
            g.set_edata("rp", "att_rp", g.edata("rp", "r")[:, 1:])
            # g.edges["rp"].data["att_rp"] = g.edges["rp"].data["r"][:, 1:]
            g.set_edata(
                "rp", "att_rp", data=g.edata("rp", "att_rp")[:, -1] + 1, index=-1
            )
            # g.edges["rp"].data["att_rp"][:, -1] = (
            #     g.edges["rp"].data["att_rp"][:, -1] + 1
            # )
            g.set_edata("rrp", "att_rp", g.edata("rrp", "r")[:, 1:])
            # g.edges["rrp"].data["att_rp"] = g.edges["rrp"].data["r"][:, 1:]
            g.set_edata(
                "rrp", "att_rp", index=-1, data=g.edata("rrp", "att_rp")[:, -1] + 1
            )
            # g.edges["rrp"].data["att_rp"][:, -1] = (
            #     g.edges["rrp"].data["att_rp"][:, -1] + 1
            # )

    if g.num_edges(etype="rc") != 0:
        g.set_edata("rc", "rid", (g.edata("rc", "rid") + 1).int())
        # g.edges["rc"].data["rid"] = (g.edges["rc"].data["rid"] + 1).int()
        g.set_edata("rc", "att_rc", index=0, data=g.edata("rc", "val"))
        # g.edges["rc"].data["att_rc"][:, 0] = g.edges["rc"].data["val"]
        g.set_edata("rc", "att_rc", index=1, data=g.edata("rc", "valr"))
        # g.edges["rc"].data["att_rc"][:, 1] = g.edges["rc"].data["valr"]

    return g, ["type", "rid", "att_rc", "att_rp"]
