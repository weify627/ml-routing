import torch
import numpy as np
import networkx as nx
from softmin_routing import *
from min_congestion import *

def create_graph(nV=12, nE=32):
    G = nx.DiGraph()
    G.add_nodes_from(range(nV))
    setE = []
    #np.random.seed(1)
    #for i in range(nV):
    #    while len(setE)%2==0:
    #        a=(i,np.random.choice(nV,1)[0])
    #        if not a in setE:
    #            setE +=[a]
    #    while len(setE)%2!=0:
    #        a=(np.random.choice(nV,1)[0],i)
    #        if not a in setE:
    #            setE +=[a]
    #while len(setE)!=nE:
    #    a=(np.random.choice(nV,1)[0],np.random.choice(nV,1)[0])
    #    if not a in setE:
    #            setE +=[a]        
    ##idx = np.random.RandomState(seed=8).choice(len(setE),32,replace=False)
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

    return G



class RouteEnv(object):
    def __init__(self, dataset, seq_len, dm_size, act_len, seq_num=7, gamma=2):
        self.action_size = (act_len,)
        self.seq_num = seq_num
        self.dm_size = dm_size
        self.state_size = seq_len* dm_size* dm_size
        self.state_shape = (seq_len, dm_size, dm_size)
        dataset.x_cyc = dataset.gen_seq(seq_len)
        self.dataset = dataset
        self.seq_len = seq_len
        self.G = create_graph(nV=dm_size)
        self.gamma = gamma

    def step(self, idx, action):
        assert idx<self.seq_len
        assert idx==(self.idx+1)
        G = nx.DiGraph(self.G)
        D = self.last_target.reshape(self.dm_size, self.dm_size)
        action = action - action.min()+0.2
        action_max = action.max()
        for k, e in enumerate(G.edges()):
            G[e[0]][e[1]]['weight'] = action[k]
            #G[e[0]][e[1]]['capacity'] = action_max*10
        #pause()
        _,_,_, m_cong = softmin_routing(G, D, gamma=self.gamma)
        _,_,opt = min_congestion(G, D)
        reward=-m_cong/opt
        self.state, self.last_target, self.idx = self.dataset.__getitem__(idx)
        done = 1 if idx == (self.seq_len-1) else 0
        return self.state.reshape(-1), reward, done, {}

    def reset(self):
        seq_id = np.random.choice(self.seq_num,1)[0]
        self.state, self.last_target, self.idx = self.dataset.__getitem__(0)
        self.G = create_graph(nV=self.dm_size)
        return np.array(self.state).reshape(-1)


dtype=torch.float64
def to_device(device , *args):
    return [x.to(device).to(dtype) for x in args]

def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns
