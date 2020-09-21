import numpy as np
from scipy.optimize import minimize, fmin_cg
import matplotlib.pyplot as plt

np.random.seed(16777216)

alpha, beta = np.random.random(), np.random.random()

x = [k / 100 for k in range(101)]
y = [alpha * x_ + beta + np.random.normal() for x_ in x]


def f1(a):
    return sum([(a[0] * x_ + a[1] - y_) ** 2 for x_, y_ in zip(x, y)])


def ff1(a):
    return sum([(a[0] * x_ + a[1] - y_) ** 2 for x_, y_ in zip(x, y)])


def f2(a):
    return sum([(a[0] / (1 + a[1] * x_) - y_) ** 2 for x_, y_ in zip(x, y)])


res1 = fmin_cg(f1, [0, 0], ff1, gtol=0.001)
print(res1)
