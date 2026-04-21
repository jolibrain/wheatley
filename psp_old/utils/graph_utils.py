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

# def all_longest_distances(graph, s, weight='weight', reverse_graph=False):
#     i_w_graph = inverse_weight(graph, weight)

#     if reverse_graph:
#         i_w_graph = i_w_graph.reverse()

#     preds,distances = nx.dijkstra_predecessor_and_distance(i_w_graph, s)
#     for key in distances:
#         distances[key] *= -1
#     return distances

def all_longest_distances(graph, s, reverse_graph=False):
        copy_graph = graph.copy()
        if reverse_graph:
            copy_graph = copy_graph.reverse()

        assert(copy_graph.in_degree(s) == 0)

        dist = dict.fromkeys(copy_graph.nodes, -float('inf'))
        dist[s] = 0
        topo_order = nx.topological_sort(copy_graph)
        for n in topo_order:
            for s in copy_graph.successors(n):
                if dist[s] < dist[n] + copy_graph.edges[n,s]['weight']:
                    dist[s] = dist[n] + copy_graph.edges[n,s]['weight']
        return dist