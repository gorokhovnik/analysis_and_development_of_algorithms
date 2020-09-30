import numpy as np
import networkx as nx
from algo_lib import kmp
import random
import string
import tracemalloc
from timeit import timeit
import matplotlib.pyplot as plt

N = 1000  # maximum number of elements

s = ''.join([np.random.choice(list(string.ascii_lowercase)) for i in range(N)])

experimental_time_kmp = []
experimental_space_kmp = []

for i in range(1, N + 1):
    cur_str = s[:i]
    cur_sub = cur_str[np.random.randint(-1, i / 2):np.random.randint(i / 2 - 1, i)]
    experimental_time_kmp += [timeit(lambda: kmp(cur_sub, cur_str), number=3)]  # measure experimental time
    tracemalloc.start()
    kmp(cur_sub, cur_str)
    experimental_space_kmp += [tracemalloc.get_traced_memory()[1]]  # measure experimental space
    tracemalloc.stop()

theoretical_one_step = sum(experimental_time_kmp[100:]) / sum(range(100, N + 1))
# calculate theoretical time
theoretical_time_kmp = [theoretical_one_step * n for n in range(N + 1)]

theoretical_one_step = sum(experimental_space_kmp[100:]) / sum(range(100, N + 1))
# calculate theoretical space
theoretical_space_kmp = [theoretical_one_step * n for n in range(N + 1)]

# create plot
plt.title('Knuth-Morris-Pratt // time', fontsize=20)
plt.xlabel('String length')
plt.ylabel('Running time')
plt.plot(experimental_time_kmp, color='red', label='experimental')
plt.plot(theoretical_time_kmp, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

# create plot
plt.title('Knuth-Morris-Pratt // Space', fontsize=20)
plt.xlabel('String length')
plt.ylabel('Space')
plt.plot(experimental_space_kmp, color='red', label='experimental')
plt.plot(theoretical_space_kmp, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()


def lq(G):
    nx.algorithms.clique.graph_number_of_cliques(G)


experimental_time_lq = []
experimental_space_lq = []

for i in range(1, N + 1):
    i_ = int(np.sqrt(i) + 1)
    G = nx.grid_graph([i_, i_])
    j = i_ ** 2 - 1
    while j > i:
        G.remove_node((j % i_, j // i_))
        j -= 1
    experimental_time_lq += [timeit(lambda: lq(G), number=3)]  # measure experimental time
    tracemalloc.start()
    G = G
    lq(G)
    experimental_space_lq += [tracemalloc.get_traced_memory()[1]]  # measure experimental space
    tracemalloc.stop()

theoretical_one_step = sum(experimental_time_lq[100:]) / sum(range(100, N + 1))
# calculate theoretical time
theoretical_time_lq = [theoretical_one_step * n for n in range(N + 1)]

theoretical_one_step = sum(experimental_space_lq[100:]) / sum(range(100, N + 1))
# calculate theoretical space
theoretical_space_lq = [theoretical_one_step * n for n in range(N + 1)]

# create plot
plt.title('Max clique (d = 4) // time', fontsize=20)
plt.xlabel('Number of nodes')
plt.ylabel('Running time')
plt.plot(experimental_time_lq, color='red', label='experimental')
plt.plot(theoretical_time_lq, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

# create plot
plt.title('Max clique (d = 4) // Space', fontsize=20)
plt.xlabel('Number of nodes')
plt.ylabel('Space')
plt.plot(experimental_space_lq, color='red', label='experimental')
plt.plot(theoretical_space_lq, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

experimental_time_lq = []
experimental_space_lq = []

N = N // 5

for i in range(1, N + 1):
    G = nx.complete_graph(i)
    experimental_time_lq += [timeit(lambda: lq(G), number=3)]  # measure experimental time
    tracemalloc.start()
    G = nx.complete_graph(i)
    lq(G)
    experimental_space_lq += [tracemalloc.get_traced_memory()[1]]  # measure experimental space
    tracemalloc.stop()

theoretical_one_step = sum(experimental_time_lq[100:]) / sum([n ** 3 for n in range(100, N + 1)])
# calculate theoretical time
theoretical_time_lq = [theoretical_one_step * n ** 3 for n in range(N + 1)]

theoretical_one_step = sum(experimental_space_lq[100:]) / sum([n ** 2 for n in range(100, N + 1)])
# calculate theoretical space
theoretical_space_lq = [theoretical_one_step * n ** 2 for n in range(N + 1)]

# create plot
plt.title('Max clique (complete) // time', fontsize=20)
plt.xlabel('Number of nodes')
plt.ylabel('Running time')
plt.plot(experimental_time_lq, color='red', label='experimental')
plt.plot(theoretical_time_lq, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

# create plot
plt.title('Max clique (complete) // Space', fontsize=20)
plt.xlabel('Number of nodes')
plt.ylabel('Space')
plt.plot(experimental_space_lq, color='red', label='experimental')
plt.plot(theoretical_space_lq, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()
