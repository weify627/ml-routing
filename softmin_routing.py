import networkx as nx


def create_example():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_weighted_edges_from([(0, 1, 2), (1, 2, 4), (0, 2, 7)])

    D = np.array([[0,2,7],
                  [0,0,2],
                  [0,0,0]])

    return G, D


def softmin_routing(G, D):
    '''
    Return a routing policy given a directed graph with weighted edges and a
    deman matrix.
    args:
        G is a networkx graph with nodes and edges with weights.
        D is a demand matrix, represented as a 2D numpy array.
    '''

    return


if __name__ == '__main__':
    G, D = create_example_graph()
    softmin_routing(G, D)
