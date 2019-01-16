from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import torch
from pdb import set_trace as pause

#def create_graph(nV=12, nE=32):
def create_graph(nV=5, nE=8):
    G = nx.DiGraph()
    G.add_nodes_from(range(nV))
    setE = []
    if 1:
        np.random.seed(1)
        for i in range(nV):
            while len(setE)%2==0:
                a=(i,np.random.choice(nV,1)[0])
                if not a in setE and a[0]!=a[1]:
                    setE +=[a]
            while len(setE)%2!=0:
                a=(np.random.choice(nV,1)[0],i)
                if not a in setE and a[0]!=a[1]:
                    setE +=[a]
        while len(setE)<=nE:
            a=(np.random.choice(nV,1)[0],np.random.choice(nV,1)[0])
            if not a in setE and a[0]!=a[1]:
                    setE +=[a]        
    else:
    #idx = np.random.RandomState(seed=8).choice(len(setE),32,replace=False)
        setE = [(0,1),(0,3),(0,5),(0,6),\
                (1,7),(1,8),\
                (2,7),(2,9),(2,11),\
                (3,0),(3,10),\
                (4,1),(4,6),\
                (5,2),(5,6),\
                (6,2),(6,5),\
                (7,3),(7,4),(7,5),(7,6),(7,10),\
                (8,4),\
                (9,1),(9,5),(9,8),(9,10),(9,11),\
                (10,4),\
                (11,0),(11,7),(11,8)]
    #setE = [setE[i] for i in idx]
    #print(setE)
    G.add_edges_from(setE)

    for e in setE:
        G[e[0]][e[1]]['capacity'] = 10
        G[e[0]][e[1]]['cost'] = 1
        G[e[0]][e[1]]['weight'] = 1

    return G

bk_G = create_graph()
#bk_G = create_graph(nV=12)
bk_edge = [ e for _,e in enumerate(bk_G.edges())]



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
    G.add_edges_from([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])

    G[0][1]['weight'] = 1
    G[0][2]['weight'] = 1
    G[1][0]['weight'] = 1
    G[1][2]['weight'] = 1
    G[2][0]['weight'] = 1
    G[2][1]['weight'] = 1

    G[0][1]['capacity'] = 100
    G[0][2]['capacity'] = 100
    G[1][0]['capacity'] = 100
    G[1][2]['capacity'] = 100
    G[2][0]['capacity'] = 100
    G[2][1]['capacity'] = 100

    G[0][1]['cost'] = 1
    G[0][2]['cost'] = 1
    G[1][0]['cost'] = 1
    G[1][2]['cost'] = 1
    G[2][0]['cost'] = 1
    G[2][1]['cost'] = 1

    D = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

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
    if s == t:
        return 0.

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

def sp3(a,b,c,w,sps): 
    return w[bk_edge.index((a,b))] + sps[b][c]
    return w[bk_edge.index((a,b))] + sps[b][c]

def split_ratio_torch(G, w, s, u, t, gamma):
    denom = torch.zeros_like(w[0])
    if s == t:
        return denom

    # sp3 is the shortest path from a to c that goes through a's immediate
    # neighbor b
    #sp3 = lambda a, b, c: w[bk_edge.index((a,b))] + sps[b][c]
    if nx.has_path(G, u,t):
        sps = 0
        p = nx.shortest_path(G,u,t,weight='weight')
        for k in range(1,len(p)):
            sps += w[bk_edge.index((p[k-1],p[k]))]
    else:
        sps = np.inf
    #num = torch.exp(-gamma *(w[bk_edge.index((s,u))]+sps[u][t])) # sp3(s, u, t,w,sps))
    num = torch.exp(-gamma *(w[bk_edge.index((s,u))]+sps)) # sp3(s, u, t,w,sps))
    #num = torch.exp(-gamma *(a.detach()+b)) # (sut2)) # (w[bk_edge.index((s,u))]+)) # sp3(s, u, t,w,sps))
    #num = torch.exp(-gamma * sp3(s, u, t))

    for v in G.neighbors(s):
        if nx.has_path(G, v, t):
            sps = 0
            p = nx.shortest_path(G,v,t,weight='weight')
            for k in range(1,len(p)):
                sps += w[bk_edge.index((p[k-1],p[k]))]
        else:
            sps = np.inf
        #denom += torch.exp(-gamma * (w[bk_edge.index((s,v))]+sps[v][t])) 
        denom += torch.exp(-gamma * (w[bk_edge.index((s,v))]+sps)) 
        #denom += torch.exp(-gamma * (w[bk_edge.index((s,v))]+spss[v][t])) 
        #denom += torch.exp(-gamma * sp3(s, v, t))
    #pause()
    # Prevent division by zero badness
    if (denom.item() == 0):
        return denom

    return num / denom

def get_shortest_paths_torch(G, w, dtype=torch.float64, device=torch.device('cuda')):
    nV = G.number_of_nodes()
    sps = torch.full((nV, nV), fill_value=np.inf,dtype=dtype, device=device)
    for i in range(nV):
        for j in range(nV):
            if nx.has_path(G, i, j):
                sps[i][j] = 0
                p = nx.shortest_path(G,i,j,weight='weight')
                for k in range(1,len(p)):
                    sps[i][j] += w[bk_edge.index((p[k-1],p[k]))]

    return sps

def get_shortest_paths_torch_old(G, w, dtype=torch.float64, device=torch.device('cuda')):
    '''TODO: This needs to be made differentiable for PyTorch's automatic
    gradient. The way to do this is to replace shortest_path_length with
    a computation of the shortest path, and to compute the shortest path length
    manually by adding the edge weights along the shortest path.'''
    nV = G.number_of_nodes()
    #sps = torch.full((nV, nV), fill_value=np.inf,dtype=dtype, device=device)
    sps = np.full((nV, nV), fill_value=np.inf) #,dtype=dtype, device=device)
    path = dict(nx.all_pairs_shortest_path(G))
    for i in range(nV):
        for j in range(nV):
            if nx.has_path(G, i, j):
                sps[i][j] = 0
                p = path[i][j]
                for k in range(1,len(p)):
                    sps[i][j] += G[p[k-1]][p[k]]['weight']
                    #sps[i][j] += w[bk_edge.index((p[k-1],p[k]))]

    return sps
