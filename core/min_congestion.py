from __future__ import print_function
import sys
import numpy as np
import gurobipy as gb
import networkx as nx
import utils


def min_congestion(G, D, hard_cap=False, verbose=False):
    '''
    Compute the multi-commodity flow which minimizes maximum link
    utilization, through linear programming.
    input parameters:
        G is a networkx graph with nodes and edges. Edges must have a
        'capacity'. Edge capacity denotes the maximum possible traffic
        utilization for an edge. It can be set as a hard or soft optimization
        constraint through the 'hard_cap' parameter. Edges may additionally
        have a 'cost' attribute used for weighting the maximum link utilization.

        D is a |V| x |V| demand matrix, represented as a 2D numpy array. |V|
        here denotes the number of vertices in the graph G.

        hard_cap is a boolean flag which determines whether edge capacities are
        treated as hard or soft optimization constraints.

        verbose is a boolean flag enabling/disabling optimizer printing.

    return values:
        f_sol is a routing policy, represented as a numpy array of size
        |V| x |V| x |E| such that f_sol[s, t, i, j] yields the amount of traffic
        from source s to destination t that goes through edge (i, j).

        l_sol is numpy array of size |E| such that l[i, j] represents the total
        amount of traffic that flows through edge (i, j) under the given flow.

        m_cong is the maximal congestion for any link weighted by cost.
        ie max_{(i, j) in E} cost[i, j] * l[i, j] / cap[i, j].
    '''
    np.fill_diagonal(D,0)
    nV = G.number_of_nodes()
    nE = G.number_of_edges()

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

    # Create variables
    f = m.addVars(V, V, arcs, obj=cost, name='flow')
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

    # Flow conservation constraints
    for s, t, u in utils.cartesian_product(V, V, V):
        d = D[int(s), int(t)]
        if u==s:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==d, 'conserv')
        elif u==t:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==-d, 'conserv')
        else:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==0, 'conserv')

    # Set objective to max-link utilization (congestion)
    max_cong = m.addVar(name='congestion')
    m.addConstrs(((cost[i,j]*l[i, j])/capacity[i,j]<=max_cong for i, j in arcs))
    m.setObjective(max_cong, gb.GRB.MINIMIZE)

    # Compute optimal solution
    m.optimize()

    # Print solution
    if m.status == gb.GRB.Status.OPTIMAL:
        f_sol = m.getAttr('x', f)
        l_sol = m.getAttr('x', l)
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

        verboseprint('\nTotal traffic through link.')
        verboseprint('l(i, j) denotes the total amount of traffic that passes'
                     ' through edge (i, j).'
        )

        for i, j in arcs:
            p = l_sol[i, j]
            if p > 0:
                verboseprint('%s -> %s: %g bytes.' % (i, j, p))

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
        verboseprint('\nERROR: Flow Optimization Failed!', file=sys.stderr)
        return None, None, None

    return f_sol, l_sol, m_cong

bug=1
if __name__ == '__main__':
    if bug:
        D = np.loadtxt("./demand.txt")
        w = np.loadtxt("./w.txt")
        capacity = np.loadtxt("./capacity.txt")
        cost = np.loadtxt("./cost.txt")
        e0 = np.loadtxt("./e0.txt")
        e1 = np.loadtxt("./e1.txt")
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        E=[]
        for i in range((e0).size):
            E +=[(e0[i],e1[i])]
        G.add_edges_from(E)
        for i,e in enumerate(E):
            G[e[0]][e[1]]['capacity']=capacity[i]
            G[e[0]][e[1]]['weight']=w[i]
            G[e[0]][e[1]]['cost']=cost[i]
        GAMMA = 2
        _,_,_ = min_congestion(G, D, verbose=True)
        exit()
    G, D = utils.create_example()
    print('Linear programming for multi-commodity flow optimization.')
    f, l, m = min_congestion(G, D, hard_cap=True, verbose=True)
