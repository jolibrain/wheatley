import networkx as nx
import sys
import matplotlib.pyplot as plt

sys.path.append(".")

from resource_flowgraph import ResourceFlowGraph

def generate_graph(rg):
    DG = nx.DiGraph()

    openNodes = []
    closeNodes = []

    labels = {}

    for n in rg.nodes:
        labels[n] = str(n)
        closeNodes.append(n)
    
    for t in rg.frontier:
        openNodes.append(t[1])
        if t[1] in closeNodes:
            closeNodes.remove(t[1])

    for i in range(len(rg.edges_source)):
        print(rg.edges_att[i])
        DG.add_edge(rg.edges_source[i],rg.edges_dest[i],weight=rg.edges_att[i])
    
    pos = nx.spring_layout(DG)
    options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}

    nx.draw_networkx_nodes(DG, pos, nodelist=openNodes, node_color="tab:red", **options)
    nx.draw_networkx_nodes(DG, pos, nodelist=closeNodes, node_color="tab:blue", **options)

    nx.draw_networkx_edges(DG, pos, width=1.0, alpha=0.5)


    edge_labels = dict([((n1, n2), d['weight'])
                    for n1, n2, d in DG.edges(data=True)])

    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, label_pos=0.5,
                             font_color='red', font_size=12, font_weight='bold')

    nx.draw_networkx_labels(DG, pos, labels, font_size=16, font_color="black")

    plt.show()