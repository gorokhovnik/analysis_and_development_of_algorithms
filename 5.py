import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from warnings import filterwarnings

filterwarnings('ignore')

np.random.seed(16777216)

Nv = 100
Ne = 200

adj_list = {}
adj_matrix = np.zeros((Nv, Nv))
n_edge = 0

while n_edge < Ne:
    i = np.random.randint(0, Nv)
    j = np.random.randint(0, Nv)
    if adj_matrix[i, j] == 0 and i != j:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
        n_edge += 1

for i in range(Nv):
    adj_list[i] = []
    for j in range(Nv):
        if adj_matrix[i, j] == 1:
            adj_list[i] += [j]

print(adj_matrix, '\n')
print(adj_list, '\n')

G = nx.from_dict_of_lists(adj_list)
nx.draw(G)
plt.show()

ttl_dfs = set()
n_comp = 0
comps = []
for i in range(Nv):
    if i not in ttl_dfs:
        dfs = [i for i in nx.algorithms.traversal.dfs_preorder_nodes(G, i)]
        ttl_dfs.update(dfs)
        comps += [dfs]
        n_comp += 1

for comp in comps:
    print(comp)
print()

fromV = np.random.randint(0, Nv)
toV = np.random.randint(0, Nv)
path = nx.shortest_path(G, fromV, toV)  # uses dijkstra, but for unweighted graph, dijkstra and BFS are similar

print(fromV, toV, len(path) - 1, '>'.join([str(v) for v in path]))
