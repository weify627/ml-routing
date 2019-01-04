import sys
import numpy as np
import gurobipy as gb
from collections import defaultdict

import softmin_routing

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


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

        m_cong is the maximal congestion for any link weighted by cost. ie
        max_{(i, j) in E} cost[i, j] * l[i, j] / cap[i, j]
    '''
    m = gb.Model('netflow')

    verboseprint = print

    if not verbose:
        verboseprint = lambda *a: None
        m.setParam('OutputFlag', False )
        m.setParam('LogToConsole', False )

    # Make string array for hashing
    V = np.array([str(i) for i in V])

    if not w:
        # If weights aren't specified, make uniform
        verboseprint('Using uniform link costs.')
        w = np.ones(len(E))

    cap, cost = {}, {}
    for k, e in enumerate(E):
        i, j = str(e[0]), str(e[1])
        cap[i, j]  =  c[k]
        cost[i, j] =  w[k]

    arcs, capacity = gb.multidict(cap)

    # Create variables
    f = m.addVars(V, V, arcs, obj=cost, name='flow')
    l = m.addVars(arcs, lb=0.0, name='tot_traf_across_link')
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
    for s, t, u in cartesian_product(V, V, V):
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
        verboseprint('\nOptimal traffic flows.')
        verboseprint('\nf_{i -> j}(s, t) denotes amount of traffic from source'
                     ' s to destination t that goes through link (i, j) in E.')
        for s, t in cartesian_product(V, V):
            for i,j in arcs:
                p = f_sol[s, t, i, j]
                if p > 0:
                    verboseprint('f_{%s -> %s}(%s, %s): %g bytes.'
                                  % (i, j, s, t, p))

        l_sol = m.getAttr('x', l)
        verboseprint('\nTotal traffic per link.')
        for i, j in arcs:
            p = l_sol[i, j]
            if p > 0:
                verboseprint('%s -> %s: %g bytes.' % (i, j, p))
        m_cong = float(max_cong.x)
        verboseprint('\nMax. weighted link util: ', format(m_cong, '.4f'))
    else:
        verboseprint('\nERROR: Flow Optimization Failed!', file=sys.stderr)
        return None, None, None
    return f_sol, l_sol, m_cong


if __name__ == '__main__':
    G, D = softmin_routing.create_exampled()
    print('Linear programming for multi-commodity flow optimization.')
    f, l, m = min_congestion(G, D, hard_cap=True, verbose=True)
