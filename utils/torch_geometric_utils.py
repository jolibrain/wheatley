import torch


def find_edge(graph, edge):
    """
    Takes a torch_geometric.data.Data graph as input, and returns the index of specified
    edge if it is in the graph. Else returns None.
    """
    edge_index = graph.edge_index
    for index, cur_edge in enumerate(edge_index.transpose(0, 1)):
        if (edge[0] == cur_edge[0] and edge[1] == cur_edge[1]) or (
            edge[1] == cur_edge[0] and edge[0] == cur_edge[1]
        ):
            return index
    return None


def add_edge(graph, edge):
    """
    Takes a torch_geometric.data.Data graph as input and returns the same graph with an
    added edge.
    Note : I think that this may be bad practice since function modifies input
    """
    if max(edge[0], edge[1]) >= graph.x.shape[0]:
        raise Exception("Index of node is out of range")
    edge_index = graph.edge_index
    edge_index = torch.cat(
        (edge_index, torch.tensor([[edge[0]], [edge[1]]])), dim=1
    )
    graph.edge_index = edge_index
    return graph


def remove_edge(graph, edge):
    """
    Takes a torch_geometric.data.Data graph as input and returns the same graph without
    the specified edge
    Note : I think this may be bad practice since function modifies input
    """
    if max(edge[0], edge[1]) >= graph.x.shape[0]:
        raise Exception("Index of node is out of range")
    index = find_edge(graph, edge)
    if index is None:
        return graph
    edge_index = graph.edge_index
    mask = torch.full_like(edge_index, True)
    mask[:, index] = False
    mask = mask > 0  # Ensure this is an array of booleans
    edge_index = edge_index[mask]
    edge_index = edge_index.view(2, -1)
    graph.edge_index = edge_index
    return graph
