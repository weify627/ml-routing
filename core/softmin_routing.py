from __future__ import print_function
import numpy as np
import sys
import gurobipy as gb
from  pdb import set_trace as pause
from utils import *
import networkx as nx

def softmin_routing_torch(G, w, D, gamma=2, hard_cap=False, verbose=False):
    np.fill_diagonal(D,0)
    _,_,g, m_cong = softmin_routing(G,D)
    nV = G.number_of_nodes()
    nE = G.number_of_edges()
    #sps = get_shortest_paths_torch(G,w)
    l = torch.zeros_like(w)
    #split_torch = torch.zeros((nE,nV),dtype=torch.float64, device=torch.device('cuda'))
    for i_e in range(nE):
        s = bk_edge[i_e][0]
        t = bk_edge[i_e][1]
        for i_v in range(nV):
            #split_torch[i_e,i_v] = split_ratio_torch(G,w,s,t,i_v,gamma,sps)
            l[i_e] += g[s,i_v] * split_ratio_torch(G,w,s,t,i_v,gamma)
        l[i_e] = l[i_e]/G[s][t]['capacity'] 
    m_cong_t = l.max()
    #print("m_cong",m_cong, m_cong_t)
    #if abs(m_cong-m_cong_t.item())<0.001: print("m_cong test passed")
    return m_cong_t

def test_soft(G,w,D,gamma=2):
    nV = G.number_of_nodes()
    nE = G.number_of_edges()
    sps = get_shortest_paths(G)
    sps_t = get_shortest_paths_torch(G,w)
    for i in range(nV):
        for j in range(nV):
            if sps[i,j]!=sps_t[i,j]: print("sps",sps[i,j],sps_t[i,j])
            #assert sps[i,j]==sps_t[i,j]
    print("sps test passed!")
    for i_e in range(nE):
        s = bk_edge[i_e][0]
        t = bk_edge[i_e][1]
        for i_v in range(nV):
            split_t = split_ratio_torch(G,w,s,t,i_v,gamma,sps)
            split = split_ratio(G,s,t,i_v,gamma,sps)
            if abs(split_t.item()-split)>0.001: print(split_t,split, i_e,i_v)
            #assert split_t.item()==split
    print("split_ratio test passed!")



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
    np.fill_diagonal(D,0)

    sps = get_shortest_paths(G)

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
    for s, t in cartesian_product(V, V):
        qs = gb.quicksum(
                g[u, t] * split_ratio(G, u, v, t, gamma, sps)
                for (u, v) in G.in_edges(s)
        )
        #pause()
        m.addConstr(
            g[s, t] == qs + D[s, t],
            'split_ratio_{}_{}'.format(s, t)
        )

    # Total commodity is sum of incoming flows plus outgoing source.
    for s, t in cartesian_product(V, V):
        m.addConstr(g[s, t] == (f.sum('*', t, '*', s) + D[s, t]))

    # Flow conservation constraints.
    for s, t, u in cartesian_product(V, V, V):
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

        for s, t in cartesian_product(V, V):
            for i,j in arcs:
                p = f_sol[s, t, i, j]
                if p > 0:
                    verboseprint('f_{%s -> %s}(%s, %s): %g bytes.'
                                  % (i, j, s, t, p))

        verboseprint('\nTotal traffic at node.')
        verboseprint('g(i, j) denotes the total amount of traffic destined for'
                     ' node j that passes through node i.'
        )

        for s, t in cartesian_product(V, V):
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
        print(D, m.status)
        np.savetxt("demand.txt",D)
        w = np.zeros(nE)
        cap = np.zeros(nE)
        cost = np.zeros(nE)
        e0= np.zeros(nE)
        e1= np.zeros(nE)
        for k, e in enumerate(G.edges()):
            w[k] = G[e[0]][e[1]]['weight']
            cap[k] = G[e[0]][e[1]]['capacity']
            cost[k] = G[e[0]][e[1]]['cost']
            e0[k] = e[0]
            e1[k] = e[1]
            print(e,G[e[0]][e[1]]['cost'],  G[e[0]][e[1]]['capacity'], G[e[0]][e[1]]['weight'])

        np.savetxt("w.txt",w)
        np.savetxt("capacity.txt",cap)
        np.savetxt("cost.txt",cost)
        np.savetxt("e0.txt",e0)
        np.savetxt("e1.txt",e1)
        pause()
        verboseprint('\nERROR: Flow Optimization Failed!', file=sys.stderr)
        return None, None, None, None

    return f_sol, l_sol, g_sol, m_cong

bug=0
if __name__ == '__main__':
    if bug:
        D = np.loadtxt("../demand.txt")
        w = np.loadtxt("../w.txt")
        capacity = np.loadtxt("../capacity.txt")
        cost = np.loadtxt("../cost.txt")
        E=[(0, 1),\
        (0, 3) ,\
        (0, 5) ,\
        (0, 6) ,\
        (1, 8) ,\
        (1, 7) ,\
        (2, 9) ,\
        (2, 11),\
        (2, 7) ,\
        (3, 0) ,\
        (3, 10),\
        (4, 1) ,\
        (4, 6) ,\
        (5, 2) ,\
        (5, 6) ,\
        (6, 2) ,\
        (6, 5) ,\
        (7, 10),\
        (7, 3) ,\
        (7, 4) ,\
        (7, 5) ,\
        (7, 6) ,\
        (8, 4) ,\
        (9, 8) ,\
        (9, 1) ,\
        (9, 10),\
        (9, 11),\
        (9, 5) ,\
        (10, 4),\
        (11, 0),\
        (11, 8),\
        (11, 7)]
        print(len(E))
        G = nx.DiGraph()
        G.add_nodes_from(range(12))
        G.add_edges_from(E)
        for i,e in enumerate(E):
            G[e[0]][e[1]]['capacity']=capacity[i]
            G[e[0]][e[1]]['weight']=w[i]
            G[e[0]][e[1]]['cost']=cost[i]
    GAMMA = 2
    #_,_,_,m_cong = softmin_routing(G, D, GAMMA, verbose=True)
    #pause()
    #G, D = create_example()
    G = create_graph(12,32) 
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
    D2 =[[0.        ,0.        ,0.        ,0.07467553,0.        ,0.\
         ,0.        ,0.        ,0.        ,0.        ,0.        ,0.        ],\
           [0.63155325,0.        ,0.07924647,0.        ,0.08720418,0.20944966\
                ,0.59397937,0.51572447,0.        ,0.14958543,0.12576671,0.        ],\
            [0.        ,0.        ,0.        ,0.56884487,0.        ,0.\
                 ,0.        ,0.45998989,0.19482636,0.        ,0.54485782,0.5525511 ],\
             [0.        ,0.        ,0.        ,0.        ,0.        ,0.\
                  ,0.13893744,0.05061309,0.        ,0.        ,0.        ,0.15526118],\
              [0.        ,0.        ,0.        ,0.        ,0.        ,0.01075095\
                   ,0.03327349,0.05564292,0.06327051,0.02016232,0.00850357,0.0522263 ],\
               [0.        ,0.10034432,0.        ,0.        ,0.10741478,0.\
                    ,0.        ,0.17650218,0.21337215,0.        ,0.        ,0.        ],\
                [0.4306906 ,0.        ,0.26318789,0.00297687,0.        ,0.45115066\
                     ,0.        ,0.        ,0.39789767,0.34677174,0.        ,0.51125304],\
                 [0.        ,0.11351631,0.        ,0.24665418,0.00171465,0.09079633\
                      ,0.14393598,0.        ,0.        ,0.        ,0.        ,0.        ],\
                  [0.        ,0.59590463,0.        ,0.        ,0.37576252,0.09494492\
                       ,0.        ,0.        ,0.        ,0.        ,0.39461373,0.        ],\
                   [0.02779211,0.        ,0.48004151,0.        ,0.1233151 ,0.\
                        ,0.24210119,0.47589273,0.17315789,0.        ,0.22064177,0.        ],\
                    [0.        ,0.00737324,0.04371237,0.02974595,0.00815677,0.\
                         ,0.01025066,0.00258558,0.00070691,0.00364967,0.        ,0.02665508],\
                     [0.        ,0.04201426,0.        ,0.01147684,0.        ,0.12298539\
                          ,0.        ,0.08973908,0.09429252,0.        ,0.06337927,0.        ]]
    #D = np.array(D)
    G[0][ 1]['weight'] = 3.28800814913
    G[0][ 3]['weight'] = 3.76661126446
    G[0][ 5]['weight'] = 2.20087756067
    G[0][ 6]['weight'] = 5.85463444651
    G[1][ 8]['weight'] = 2.39048023933
    G[1][ 7]['weight'] = 5.86071104109
    G[2][ 9]['weight'] = 4.08071583167
    G[2][ 11]['weight'] = 4.6359445035
    G[2][ 7]['weight'] = 2.29757704213
    G[3][ 0]['weight'] = 4.82395352231
    G[3][ 10]['weight'] = 5.02364727537
    G[4][ 1]['weight'] = 3.24236538452
    G[4][ 6]['weight'] = 4.24308228027
    G[5][ 2]['weight'] = 4.00159475375
    G[5][ 6]['weight'] = 6.03039194205
    G[6][ 2]['weight'] = 4.36332869334
    G[6][ 5]['weight'] = 5.48024709335
    G[7][ 10]['weight'] = 4.02345098485
    G[7][ 3]['weight'] = 2.90215409206
    G[7][ 4]['weight'] = 3.26726909324
    G[7][ 5]['weight'] = 4.30541960772
    G[7][ 6]['weight'] = 3.9939494995
    G[8][ 4]['weight'] = 0.506047611233
    G[9][ 8]['weight'] = 4.38215269704
    G[9][ 1]['weight'] = 4.05885254022
    G[9][ 10]['weight'] = 4.2463743546
    G[9][ 11]['weight'] = 2.82527709364
    G[9][ 5]['weight'] = 0.2
    G[10][ 4]['weight'] = 4.83718017961
    G[11][ 0]['weight'] = 3.97261233893
    G[11][ 8]['weight'] = 4.2321296212
    G[11][ 7]['weight'] = 4.30096320225

    #for k, e in enumerate(G.edges()):
    #    G[e[0]][e[1]]['weight']=a[k]
    #print(D, D.shape) 
    w = torch.zeros(32,dtype=torch.float64, device=torch.device('cuda'))
    for i,e in enumerate(G.edges()):
        if i!=bk_edge.index((e[0],e[1])):
            print(i,e)
        w[bk_edge.index((e[0],e[1]))] = G[e[0]][e[1]]['weight']
    test_soft(G,w,D)
    m_cong_t = softmin_routing_torch(G,w, D, GAMMA, verbose=True)
    #_,_,_,m_cong = softmin_routing(G, D, GAMMA, verbose=True)
    
    pause()
    print(D) 
    for k, e in enumerate(G.edges()):
        print(e, G[e[0]][e[1]]['capacity'], G[e[0]][e[1]]['weight'])

