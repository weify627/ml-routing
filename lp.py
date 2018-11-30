import numpy as np
import gurobipy as gb
from collections import defaultdict


def get_example():
    V = np.array([0,1,2])
    E = np.array([[0,1],
                  [1,2],
                  [0,2]])
    c = np.array([5,5,10])
    D = np.array([[0,0,4],
                  [0,0,2],
                  [0,0,0]])
    return V, E, c, D


def min_congestion(V, E, c, D, w=None, hard_cap=True):
    '''
    Compute the multi-commodity flow which minimizes maximum link
    utilization, through linear programming.
    Arguments:
        V is an array of graph vertex names, ie V = [0, 1, 2]

        E is an array of directed edges, where each component of E is
        a an array [i, j] representing an edge V[i] to V[j].

        c is an array of size |E| of maximum capacity for each edge.

        D is a demand matrix, where D(i, j) represents the traffic demand
        in bytes from V[i] to V[j]

        w is an array of size |E| indicating desired edge weights (lengths)
        to be used for optimiization. If not specified, uniform weighting is
        used.

        hard_cap is a boolean flag indicating whether to make link capacity a
        hard optimization constraint or not.
    '''
    m = gb.Model('netflow')

    # Make string array for hashing
    V = [str(i) for i in V]

    if not w:
        # If weights aren't specified, make uniform
        w = np.ones(len(E))

    cap, cost = {}, {}
    for k, e in enumerate(E):
        i, j = str(e[0]), str(e[1])
        cap[i, j]  =  c[k]
        cost[i, j] =  w[k]

    arcs, capacity = gb.multidict(cap)

    # Create variables
    f = m.addVars(V, V, arcs, obj=cost, name='flow')
    l = m.addVars(arcs, lb=0.0, name='link util')
    m.addConstrs((l[i, j] == f.sum('*', '*', i, j)
                  for i, j in arcs), 'link util is sum traffic')

    # Arc capacity constraints
    if hard_cap:
        m.addConstrs((l[i, j] <= capacity[i,j]
                      for i, j in arcs), "util capacity")

    # Flow conservation constraints
    for s in V:
        for t in V:
            for u in V:
                d = D[int(s), int(t)]
                if (u==s):
                    m.addConstr((f.sum(s, t, u, '*') -
                                 f.sum(s, t, '*', u) ==
                                 d), 'conserv')
                elif (u==t):
                    m.addConstr(f.sum(s, t, u, '*') -
                                 f.sum(s, t, '*', u) ==
                                 -d, 'conserv')
                else:
                    m.addConstr(f.sum(s, t, u, '*') -
                                 f.sum(s, t, '*', u) ==
                                 0, 'conserv')

    # Set objective to max-link utilization (congestion)
    max_cong = m.addVar(name='congestion')
    m.addConstrs((l[i, j]/capacity[i,j] <= max_cong
                  for i, j in arcs))
    m.setObjective(max_cong, gb.GRB.MINIMIZE)

    # Compute optimal solution
    m.optimize()

    # Print solution
    if m.status == gb.GRB.Status.OPTIMAL:
        solution = m.getAttr('x', l)
        print('\nOptimal flows for %s:' % 'bytes')
        for i,j in arcs:
            if solution[i,j] > 0:
                print('%s -> %s: %g' % (i, j, solution[i,j]))
    return


if __name__ == '__main__':
    V, E, c, D = get_example()
    print('Linear programming for multi-commodity flow optimization')
    min_congestion(V, E, c, D)
