import numpy as np
from gurobipy import *


def get_example():
    V = np.array([0,1,2])
    E = np.array([[1,2],
                  [2,3],
                  [1,3]])
    c = np.array([5,5,10])
    D = np.array([[0,0,3],
                  [0,0,0],
                  [0,0,0]])
    return V, E, c, D


def min_congestion(V, E, c, D):
    return


if __name__ == '__main__':
    V, E, c, D = get_example()
    print('Linear programming for multi-commodity flow optimization')
    min_congestion(V, E, c, D)
