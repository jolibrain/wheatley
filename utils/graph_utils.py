import networkx as nx

def inverse_weight(graph, weight='weight'):
    copy_graph = graph.copy()
    for u,v in copy_graph.edges():
        copy_graph[u][v]['weight'] = copy_graph[u][v]['weight']*-1 
    return copy_graph

def longest_path(graph, s, t, weight='weight'):
    i_w_graph = inverse_weight(graph, weight)
    path = nx.dijkstra_path(i_w_graph, s, t)
    return path

def all_longest_distances(graph, s, weight='weight', reverse_graph=False):
    i_w_graph = inverse_weight(graph, weight)

    if reverse_graph:
        i_w_graph = i_w_graph.reverse()

    preds,distances = nx.dijkstra_predecessor_and_distance(i_w_graph, s)
    for key in distances:
        distances[key] *= -1
    return distances