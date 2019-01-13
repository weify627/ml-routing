from __future__ import print_function
import numpy as np
import sys
import gurobipy as gb
from  pdb import set_trace as pause
import utils
import networkx as nx
import sys

def create_graph(nV=12, nE=32):
    G = nx.DiGraph()
    G.add_nodes_from(range(nV))
    setE = []
    np.random.seed(1)
    if 0:
        for i in range(nV):
            for j in range(i+1,nV):
                setE +=[(i,j)]
        idx = np.random.RandomState(seed=1).choice(len(setE),32,replace=False)
        setE = [setE[i] for i in idx]
    else:
        for i in range(nV):
            while len(setE)%2==0:
                a=(i,np.random.choice(nV,1)[0])
                if (not a in setE) and a[0]!=a[1]:
                    setE +=[a]
            while len(setE)%2!=0:
                a=(np.random.choice(nV,1)[0],i)
                if (not a in setE) and a[0]!=a[1]:
                    setE +=[a]
        while len(setE)!=nE:
            a=(np.random.choice(nV,1)[0],np.random.choice(nV,1)[0])
            if (not a in setE) and a[0]!=a[1]:
                    setE +=[a]        
    #idx = np.random.RandomState(seed=8).choice(len(setE),32,replace=False)
    #setE = [setE[i] for i in idx]
    print(setE)
    G.add_edges_from(setE)

    for e in setE:
        G[e[0]][e[1]]['capacity'] = 10
        G[e[0]][e[1]]['cost'] = 1
        #G[e[0]][e[1]]['weight'] = 1
    a=np.ones((nV,nV))
    #return G, (a-np.eye(nV)).T
    return G, (a-np.triu(a)).T


def softmin_routing(G, D, gamma=2, hard_cap=False, verbose=False):
    '''
    Return a routing policy given a directed graph with weighted edges and a
    demand matrix.
    input parameters:
        G is a networkx graph with nodes and edges. Edges must have both a
        'capacity' attribute and a 'weight' attribute. Edge capacity denotes the
        maximum possible traffic utilization for an edge. It can be set as a
        hard or soft optimization constraint through the 'hard_cap' parameter.
        The edge 'weight' attribute is used for determining shortest paths.
        Edges may additionally have a 'cost' attribute used for weighting the
        maximum link utilization.

        D is a |V| x |V| demand matrix, represented as a 2D numpy array. |V|
        here denotes the number of vertices in the graph G.

        gamma is floating point number used as a parameter for the softmin
        function (exponential scaling). The larger the value for gamma, the
        closer the method is to shortest path routing.

        hard_cap is a boolean flag which determines whether edge capacities are
        treated as hard or soft optimization constraints.

        verbose is a boolean flag enabling/disabling optimizer printing.

    return values:
        f_sol is a routing policy, represented as a numpy array of size
        |V| x |V| x |E| such that f_sol[s, t, i, j] yields the amount of traffic
        from source s to destination t that goes through edge (i, j).

        l_sol is numpy array of size |E| such that l[i, j] represents the total
        amount of traffic that flows through edge (i, j) under the given flow.

        g_sol is a numpy array of size |V| x |V| such that g(i, j) is the total
        amount of traffic destined for node j that ever arrives at node i, which
        includes the inital demand from i to j.

         m_cong is the maximal congestion for any link weighted by cost.
        ie max_{(i, j) in E} cost[i, j] * l[i, j] / cap[i, j].
    '''
    nV = G.number_of_nodes()
    nE = G.number_of_edges()

    sps = utils.get_shortest_paths(G)

    m = gb.Model('netflow')

    verboseprint = print

    if not verbose:
        verboseprint = lambda *a: None
        m.setParam('OutputFlag', False )
        m.setParam('LogToConsole', False )

    V = np.array([i for i in G.nodes()])

    cost = {}
    for k, e in enumerate(G.edges()):
        if 'cost' in G[e[0]][e[1]]:
            cost[e] = G[e[0]][e[1]]['cost']
        else:
            # If costs aren't specified, make uniform.
            cost[e] = 1.0

    cap = {}
    for k, e in enumerate(G.edges()):
        cap[e]  =  G[e[0]][e[1]]['capacity']

    arcs, capacity = gb.multidict(cap)
    #pause()
    # Create variables.
    f = m.addVars(V, V, arcs, lb=0.0, name='flow')
    g = m.addVars(V, V, lb=0.0, name='traf_at_node')
    l = m.addVars(arcs, lb=0.0, name='tot_traf_across_link')

    # Link utilization is sum of flows.
    m.addConstrs(
            (l[i, j] == f.sum('*', '*', i, j) for i, j in arcs),
            'l_sum_traf',
            )

    # Arc capacity constraints
    if hard_cap:
        verboseprint('Capacity constraints set as hard constraints.')
        m.addConstrs(
            (l[i, j] <= capacity[i,j] for i, j in arcs),
            'traf_below_cap',
            )

    # Total commodity at node is sum of incoming commodities times split
    # ratios plus the source demand.
    for s, t in utils.cartesian_product(V, V):
        qs = gb.quicksum(
                g[u, t] * utils.split_ratio(G, u, v, t, gamma, sps)
                for (u, v) in G.in_edges(s)
        )
        #pause()
        m.addConstr(
            g[s, t] == qs + D[s, t],
            'split_ratio_{}_{}'.format(s, t)
        )

    # Total commodity is sum of incoming flows plus outgoing source.
    for s, t in utils.cartesian_product(V, V):
        m.addConstr(g[s, t] == (f.sum('*', t, '*', s) + D[s, t]))

    # Flow conservation constraints.
    for s, t, u in utils.cartesian_product(V, V, V):
        d = D[int(s), int(t)]
        if u==s:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==d, 'conserv')
        elif u==t:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==-d, 'conserv')
        else:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==0, 'conserv')

    # Compute max-link utilization (congestion).
    max_cong = m.addVar(name='congestion')
    m.addConstrs(((cost[i,j]*l[i, j])/capacity[i,j]<=max_cong for i, j in arcs))

    # Compute optimal solution.
    m.optimize()
    # Print solution.
    if m.status == gb.GRB.Status.OPTIMAL:
        l_sol = m.getAttr('x', l)
        g_sol = m.getAttr('x', g)
        f_sol = m.getAttr('x', f)
        m_cong = float(max_cong.x)

        verboseprint('\nOptimal traffic flows.')
        verboseprint('f_{i -> j}(s, t) denotes amount of traffic from source'
                     ' s to destination t that goes through link (i, j) in E.')

        for s, t in utils.cartesian_product(V, V):
            for i,j in arcs:
                p = f_sol[s, t, i, j]
                if p > 0:
                    verboseprint('f_{%s -> %s}(%s, %s): %g bytes.'
                                  % (i, j, s, t, p))

        verboseprint('\nTotal traffic at node.')
        verboseprint('g(i, j) denotes the total amount of traffic destined for'
                     ' node j that passes through node i.'
        )

        for s, t in utils.cartesian_product(V, V):
            p = g_sol[s, t]
            if p > 0:
                verboseprint('g({}, {}): {} bytes.'.format(s, t, p))

        verboseprint('\nTotal traffic through link.')
        verboseprint('l(i, j) denotes the total amount of traffic that passes'
                     ' through edge (i, j).'
        )

        for i, j in arcs:
            p = l_sol[i, j]
            if p > 0:
                verboseprint('l({}, {}): {} bytes.'.format(i, j, p))

        verboseprint('\nMaximum weighted link utilization (or congestion):',
                     format(m_cong, '.4f')
                     )

    else:
        print(D) 
        for k, e in enumerate(G.edges()):
            print(e, G[e[0]][e[1]]['capacity'], G[e[0]][e[1]]['weight'])
        #pause()
        verboseprint('\nERROR: Flow Optimization Failed!', file=sys.stderr)
        return None, None, None, None

    return f_sol, l_sol, g_sol, m_cong


if __name__ == '__main__':
    GAMMA = 2
    #G, D = utils.create_example()
    G, D = create_graph(12,32) 
    D =[  [0.        , 0.        , 0.        , 0.07467553, 0.        , 0.      , \
        0.        , 0.        , 0.        , 0.        , 0.        , 0.        ], \
       [0.63155325, 0.        , 0.07924647, 0.        , 0.08720418, 0.20944966,\
        0.59397937, 0.51572447, 0.        , 0.14958543, 0.12576671, 0.        ], \
       [0.        , 0.        , 0.        , 0.56884487, 0.        , 0.,\
        0.        , 0.45998989, 0.19482636, 0.        , 0.54485782, 0.5525511 ], \
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.,\
        0.13893744, 0.05061309, 0.        , 0.        , 0.        , 0.15526118], \
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.01075095,\
        0.03327349, 0.05564292, 0.06327051, 0.02016232, 0.00850357, 0.0522263 ], \
       [0.        , 0.10034432, 0.        , 0.        , 0.10741478, 0.,\
        0.        , 0.17650218, 0.21337215, 0.        , 0.        , 0.        ], \
       [0.4306906 , 0.        , 0.26318789, 0.00297687, 0.        , 0.45115066,\
        0.        , 0.        , 0.39789767, 0.34677174, 0.        , 0.51125304], \
       [0.        , 0.11351631, 0.        , 0.24665418, 0.00171465, 0.09079633,\
        0.14393598, 0.        , 0.        , 0.        , 0.        , 0.        ], \
       [0.        , 0.59590463, 0.        , 0.        , 0.37576252, 0.09494492,\
        0.        , 0.        , 0.        , 0.        , 0.39461373, 0.        ], \
       [0.02779211, 0.        , 0.48004151, 0.        , 0.1233151 , 0.,\
        0.24210119, 0.47589273, 0.17315789, 0.        , 0.22064177, 0.        ], \
       [0.        , 0.00737324, 0.04371237, 0.02974595, 0.00815677, 0.,\
        0.01025066, 0.00258558, 0.00070691, 0.00364967, 0.        , 0.02665508], \
       [0.        , 0.04201426, 0.        , 0.01147684, 0.        , 0.12298539, \
        0.        , 0.08973908, 0.09429252, 0.        , 0.06337927, 0.        ]]
    D = np.array(D)
   # (0, 1) 10 1.80491612079
   # (0, 3) 10 3.7873641032
   # (0, 5) 10 1.92617650076
   # (0, 6) 10 3.82702906394
   # (1, 8) 10 2.83159417357
   # (1, 7) 10 2.79699952342
   # (2, 9) 10 2.83729896846
   # (2, 11) 10 3.98279304282
   # (2, 7) 10 0.918321261224
   # (3, 0) 10 2.98400036154
   # (3, 10) 10 3.53005968328
   # (4, 1) 10 0.2
   # (4, 6) 10 2.24233989062
   # (5, 2) 10 3.15088420068
   # (5, 6) 10 1.65351821018
   # (6, 2) 10 3.56747206162
   # (7, 10) 10 3.30538146296
   # (7, 4) 10 2.99235330941
   # (7, 5) 10 2.9396871552
   # (7, 6) 10 2.00831815541
   # (8, 8) 10 3.1490433213
   # (8, 4) 10 2.90105821246
   # (9, 1) 10 3.91300169873
   # (9, 5) 10 0.733227312389
   # (9, 8) 10 1.58772447741
   # (9, 9) 10 1.42906939533
   # (9, 10) 10 2.22005556027
   # (9, 11) 10 2.06942632645
   # (10, 4) 10 1.31760122518
   # (11, 0) 10 4.06956464006
   # (11, 8) 10 4.09511464017
   # (11, 7) 10 2.59341397407
    a=(1.8049161207, \
    3.7873641037, \
    1.9261765007, \
    3.8270290637, \
    2.8315941737, \
    2.7969995237, \
    2.8372989687, \
    3.9827930427, \
    0.9183212617, \
    2.9840003617, \
    3.5300596837, \
    0.2,\
    2.2423398907, \
    3.1508842007, \
    1.6535182107, \
    3.5674720617, \
    3.3053814627, \
    2.9923533097, \
    2.9396871557, \
    2.0083181557, \
    3.1490433217, \
    2.9010582127, \
    3.9130016987, \
    0.7332273127, \
    1.5877244777, \
    1.4290693957, \
    2.2200555607, \
    2.0694263267, \
    1.3176012257, \
    4.0695646407, \
    4.0951146407, \
    2.5934139747)

    for k, e in enumerate(G.edges()):
        G[e[0]][e[1]]['weight']=a[k]
    print(D, D.shape) 
    softmin_routing(G, D, GAMMA, verbose=True)
    print(D) 
    for k, e in enumerate(G.edges()):
        print(e, G[e[0]][e[1]]['capacity'], G[e[0]][e[1]]['weight'])

