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


def min_congestion(V, E, c, D, w=None):
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
    '''
    m = gb.Model('netflow')
    commodities = ['bytes']

    if not w:
        # If weights aren't specified, make uniform
        w = np.ones(len(E))

    cap, cost, inflow = {}, {}, defaultdict(lambda:0, {})
    for h in commodities:
        for k, e in enumerate(E):
            i, j = str(e[0]), str(e[1])
            cap[i, j]      =  c[k]
            cost[h, i, j]  =  w[k]
            inflow[h, i]  +=  D[int(i), int(j)]
            inflow[h, j]  -=  D[int(i), int(j)]

    arcs, capacity = gb.multidict(cap)

    # Create variables
    flow = m.addVars(commodities, arcs, obj=cost, name="flow")

    # Arc capacity constraints
    m.addConstrs(
	(flow.sum('*',i,j) <= capacity[i,j] for i,j in arcs), "cap")

    # Flow conservation constraints
    V = [str(i) for i in V]
    m.addConstrs(
	(flow.sum(h,'*',j) + inflow[h,j] == flow.sum(h,j,'*')
	for h in commodities for j in V), "node")

    # Set objective to max-link utilization (congestion)
    max_cong = m.addVar(lb=0.0, obj=1.0, name='congestion')
    m.addConstrs((flow[h, i, j]/capacity[i,j] <= max_cong
                  for h in commodities
                  for i, j in arcs))
    m.setObjective(max_cong, gb.GRB.MINIMIZE)

    # Compute optimal solution
    m.optimize()

    # Print solution
    if m.status == gb.GRB.Status.OPTIMAL:
        solution = m.getAttr('x', flow)
        for h in commodities:
            print('\nOptimal flows for %s:' % h)
            for i,j in arcs:
                if solution[h,i,j] > 0:
                    print('%s -> %s: %g' % (i, j, solution[h,i,j]))
    return


if __name__ == '__main__':
    V, E, c, D = get_example()
    print('Linear programming for multi-commodity flow optimization')
    min_congestion(V, E, c, D)
