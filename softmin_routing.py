import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def create_example():
    G = nx.DiGraph()
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
                           width=3,
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


def _

    return


def softmin_routing(G, D, gamma=2):
    '''
    Return a routing policy given a directed graph with weighted edges and a
    deman matrix.
    args:
        G is a networkx graph with nodes and edges with weights.

        D is a V x V demand matrix, represented as a 2D numpy array.

        gamma is a parameter for the softmin function (exponential scaling).
        The larger the value for gamma, the closer the method is to shortest
        path routing.

    return vals:
        F is the V x V x E routing policy that yields for each
        source-destination pair, the amount of traffic that flows through edge
        e.
    '''
    nV = G.number_of_nodes()
    nE = G.number_of_edges()
    F = np.zeros((nV, nV, nV, nV))

    for i in range(nV):
        for j in range(nV):
            if D[i, j]:
                sp = []

                for k, l in enumerate(G.neighbors(i)):
                    w_il = G[i][l]['weight']

                    if nx.has_path(G, l, j):
                        splj = nx.shortest_path_length(G, l, j, weight='weight')
                        sp.append((l, splj + w_il))
                    else:
                        sp.append((l, 'inf'))

                denom = 0
                denom = sum([np.exp(-gamma * t[1]) if (t[1]!='inf') else 0
                             for t in sp]
                             )

                for tup in sp:
                    e = (i, tup[0])
                    w = np.exp(-gamma * tup[1])
                    F[i][j][i][l] = w / denom

    print(F)

    return


if __name__ == '__main__':
    G, D = create_example()
    softmin_routing(G, D)
