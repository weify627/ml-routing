import sys
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

import networkx as nx
import gurobipy as gb

import lp



def create_example():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_weighted_edges_from([(0, 1, 2), (1, 2, 4), (0, 2, 7)])

    D = np.array([[0,2,3],
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


def split_ratio(G, s, u, t, gamma, sps):
    sp3 = lambda a, b, c: G[a][b]['weight'] + sps[b][c]
    num = np.exp(-gamma * sp3(s, u, t))

    denom = 0.0
    for v in G.neighbors(s):
        denom += np.exp(-gamma * sp3(s, v, t))

    if (denom == 0):
        return 0.

    return num / denom


def get_shortest_paths(G):
    nV = G.number_of_nodes()
    sps = np.full((nV, nV), fill_value=np.inf)

    for i in range(nV):
        for j in range(nV):
            if nx.has_path(G, i, j):
                sps[i][j] = nx.shortest_path_length(G, i, j, weight='weight')

    return sps


def softmin_routing(G, D, gamma=2, verbose=False):
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
    sps = get_shortest_paths(G)

    m = gb.Model('netflow')

    verboseprint = print

    if not verbose:
        verboseprint = lambda *a: None
        m.setParam('OutputFlag', False )
        m.setParam('LogToConsole', False )

    # Make string array for hashing
    V = np.array([i for i in G.nodes()])
    arcs = gb.tuplelist(G.edges())

    # Create variables
    f = m.addVars(V, V, arcs, lb=0.0, name='flow')
    g = m.addVars(V, V, lb=0.0, name='traf_at_node')

    for s, t in lp.cartesian_product(V, V):
        qs = gb.quicksum(
                g[u, t]*split_ratio(G, u, v, t, gamma, sps)
                for (u, v) in G.in_edges(s)
        )
        m.addConstr(
            g[s, t] == qs + D[s, t],
            'split_ratio_{}_{}'.format(s, t)
        )

    for s, t in lp.cartesian_product(V, V):
        m.addConstr(g[s, t] == (f.sum('*', t, '*', s) + D[s, t]))

    # Flow conservation constraints
    for s, t, u in lp.cartesian_product(V, V, V):
        d = D[int(s), int(t)]
        if u==s:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==d, 'conserv')
        elif u==t:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==-d, 'conserv')
        else:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==0, 'conserv')

    # Set objective to max-link utilization (congestion)
    #max_cong = m.addVar(name='congestion')
    # m.addConstrs(((cost[i,j]*l[i, j])/capacity[i,j]<=max_cong for i, j in arcs))

    # Compute optimal solution
    m.optimize()

    # Print solution
    if m.status == gb.GRB.Status.OPTIMAL:
        f_sol = m.getAttr('x', f)
        verboseprint('\nOptimal traffic flows.')
        verboseprint('\nf_{i -> j}(s, t) denotes amount of traffic from source'
                     ' s to destination t that goes through link (i, j) in E.')
        for s, t in lp.cartesian_product(V, V):
            for i,j in arcs:
                p = f_sol[s, t, i, j]
                if p >= 0:
                    verboseprint('f_{%s -> %s}(%s, %s): %g bytes.'
                                  % (i, j, s, t, p))

        g_sol = m.getAttr('x', g)
        verboseprint('\nTotal traffic at node.')
        for s, t in lp.cartesian_product(V, V):
            p = g_sol[s, t]
            if p >= 0:
                verboseprint('g({}, {}): {} bytes.'.format(s, t, p))
        #m_cong = float(max_cong.x)
        #verboseprint('\nMax. weighted link util: ', format(m_cong, '.4f'))
    else:
        verboseprint('\nERROR: Flow Optimization Failed!', file=sys.stderr)
        return None, None, None
    return f_sol#, l_sol, m_cong


if __name__ == '__main__':
    G, D = create_example()
    softmin_routing(G, D, gamma=2, verbose=True)
