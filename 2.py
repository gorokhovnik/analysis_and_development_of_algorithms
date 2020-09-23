import numpy as np
import matplotlib.pyplot as plt
from algo_lib import exhaustive_search_1dim, dichotomy, golden_section, exhaustive_search_2dim, gauss, nelder_mead

np.random.seed(16777216)

'''
first task
'''


def f11(x):
    return x ** 3


def f12(x):
    return abs(x - 0.2)


def f13(x):
    return x * np.sin(1 / x)


print(exhaustive_search_1dim(f11))
print(dichotomy(f11))
print(golden_section(f11))
print(exhaustive_search_1dim(f12))
print(dichotomy(f12))
print(golden_section(f12))
print(exhaustive_search_1dim(f13, 0.01))
print(dichotomy(f13, 0.01))
print(golden_section(f13, 0.01), '\n' * 8)

'''
second task
'''

alpha, beta = np.random.random(), np.random.random()

x = [k / 100 for k in range(101)]
y = [alpha * x_ + beta + np.random.normal() for x_ in x]


def f21(a, b):
    return sum([(a * x_ + b - y_) ** 2 for x_, y_ in zip(x, y)])


def f22(a, b):
    return sum([(a / (1 + b * x_) - y_) ** 2 for x_, y_ in zip(x, y)])


es1 = exhaustive_search_2dim(f21)
print(es1)
g1 = gauss(f21)
print(g1)
nm1 = nelder_mead(f21)
print(nm1)
es2 = exhaustive_search_2dim(f22)
print(es2)
g2 = gauss(f22)
print(g2)
nm2 = nelder_mead(f22)
print(nm2)

plt.title('Exhaustive search // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([es1[2][0] * x_ + es1[2][1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Exhaustive search // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([es2[2][0] / (1 + es2[2][1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Gauss // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([g1[2][0] * x_ + g1[2][1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Gauss // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([g2[2][0] / (1 + g2[2][1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Nelder-Mead // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([nm1[2][0] * x_ + nm1[2][1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Nelder_Mead // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([nm2[2][0] / (1 + nm2[2][1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()
