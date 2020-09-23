import numpy as np
import networkx as nx
from timeit import timeit
import matplotlib.pyplot as plt
from warnings import filterwarnings

filterwarnings('ignore')

np.random.seed(16777216)

'''
1 task
'''

Nv = 100
Ne = 500

adj_list = {}
adj_matrix = np.zeros((Nv, Nv))
n_edge = 0

while n_edge < Ne:
    i = np.random.randint(0, Nv)
    j = np.random.randint(0, Nv)
    if adj_matrix[i, j] == 0 and i != j:
        adj_matrix[i, j] = np.random.randint(1, 100)
        adj_matrix[j, i] = adj_matrix[i, j]
        n_edge += 1

G = nx.from_numpy_matrix(adj_matrix)

fromV = np.random.randint(0, Nv)
dijkstra_time = timeit(lambda: nx.shortest_path(G, fromV, method='dijkstra'), number=10)
bellman_ford_time = timeit(lambda: nx.shortest_path(G, fromV, method='bellman-ford'), number=10)

print(dijkstra_time, bellman_ford_time, '\n\n')

'''
2 task
'''

NV = 10
Nv = NV ** 2 - 30


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


G = nx.grid_graph(dim=[NV, NV])

n_ = NV ** 2
while n_ > Nv:
    i = np.random.randint(0, NV)
    j = np.random.randint(0, NV)
    if (i, j) in G.nodes:
        G.remove_node((i, j))
        n_ -= 1

fromV = list(G.nodes)[np.random.randint(0, len(G.nodes))]
toV = list(G.nodes)[np.random.randint(0, len(G.nodes))]
path = nx.astar_path(G, fromV, toV, dist)
print(fromV, toV, len(path) - 1, '>'.join([str(v) for v in path]), '\n\n')

fromV = list(G.nodes)[np.random.randint(0, len(G.nodes))]
toV = list(G.nodes)[np.random.randint(0, len(G.nodes))]
path = nx.astar_path(G, fromV, toV, dist)
print(fromV, toV, len(path) - 1, '>'.join([str(v) for v in path]))

dijkstra_time = timeit(lambda: nx.shortest_path(G, fromV, toV, method='dijkstra'), number=5)
astar_time = timeit(lambda: nx.astar_path(G, fromV, toV, dist), number=5)

print(dijkstra_time, astar_time)
