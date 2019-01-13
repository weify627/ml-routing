from matplotlib import pyplot as plt
import networkx as nx
import numpy as np


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def create_example():
    '''Creates an example graph with some weights, capacities, and costs on its
    edges, together with a demand matrix. This is supposed to show what the
    input to the optimization functions looks like.'''
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])

    G[0][1]['weight'] = 2
    G[1][2]['weight'] = 4
    G[0][2]['weight'] = 7

    G[0][1]['capacity'] = 5
    G[1][2]['capacity'] = 5
    G[0][2]['capacity'] = 10

    G[0][1]['cost'] = 1
    G[1][2]['cost'] = 1
    G[0][2]['cost'] = 1

    D = np.array([[0,2,7],
                  [0,0,3],
                  [0,0,0]])

    return G, D


def draw_graph(G):
    '''Utility for drawing a graph G and its weights as a plt diagram.'''
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


def split_ratio(G, s, u, t, gamma, sps):
    '''Computes the fraction of the traffic at s which is destined for t that
    will go through s's immediate neighbor u, through the softmin function with
    parameter gamma.
    input parameters:
        G is a networkx graph with weighted edges.

        s, u, and t are vertex indices (integers) in G.

        gamma is the scaling factor for the exponent with base e.

        sps is a |V| x |V| np array such that sps[i, j] yields the length of
        the shortest path from vertex i to vertex j in G. Note that if there is
        no path from i to j in G, sps[i, j] = np.inf, a numpy value representing
        infinity which behaves well with the exponential. That is,
        np.exp(-gamma * np.inf) = 0.

    return values:
        returns a floating point representing the fraction of traffic at s which
        is destined for t that will go through s's immediate neighbor u as
        prescribed by softmin routing with parameter gamma.
    '''
    # sp3 is the shortest path from a to c that goes through a's immediate
    # neighbor b
    sp3 = lambda a, b, c: G[a][b]['weight'] + sps[b][c]

    num = np.exp(-gamma * sp3(s, u, t))

    denom = 0.0
    for v in G.neighbors(s):
        denom += np.exp(-gamma * sp3(s, v, t))

    # Prevent division by zero badness
    if (denom == 0):
        return 0.

    return num / denom


def get_shortest_paths(G):
    '''TODO: This needs to be made differentiable for PyTorch's automatic
    gradient. The way to do this is to replace shortest_path_length with
    a computation of the shortest path, and to compute the shortest path length
    manually by adding the edge weights along the shortest path.'''
    nV = G.number_of_nodes()
    sps = np.full((nV, nV), fill_value=np.inf)

    for i in range(nV):
        for j in range(nV):
            if nx.has_path(G, i, j):
                sps[i][j] = nx.shortest_path_length(G, i, j, weight='weight')

    return sps


