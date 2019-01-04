import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def create_example():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_weighted_edges_from([(0, 1, 2), (1, 2, 4), (0, 2, 7)])

    D = np.array([[0,2,7],
                  [0,0,2],
                  [0,0,0]])

    return G, D


def draw_graph(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=700)

    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist=esmall,
                           width=6,
                           alpha=0.5,
                           edge_color='b',
                           style='dashed'
                           )

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=20)

    plt.axis('off')
    plt.show()
    return


def softmin_routing(G, D):
    '''
    Return a routing policy given a directed graph with weighted edges and a
    deman matrix.
    args:
        G is a networkx graph with nodes and edges with weights.

        D is a demand matrix, represented as a 2D numpy array.

    return vals:
        F is the V x V x E routing policy that yields for each
        source-destination pair, the amount of traffic that flows through edge
        e.
    '''
    print(D)
    draw_graph(G)
    return


if __name__ == '__main__':
    G, D = create_example()
    softmin_routing(G, D)
