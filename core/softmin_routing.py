import numpy as np
import gurobipy as gb
from  pdb import set_trace as pause
import utils


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

    verboseprint = 0 #print

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
        pause()
        verboseprint('\nERROR: Flow Optimization Failed!', file=sys.stderr)
        return None, None, None, None

    return f_sol, l_sol, g_sol, m_cong


if __name__ == '__main__':
    GAMMA = 2
    G, D = utils.create_example()
    softmin_routing(G, D, GAMMA, verbose=True)

