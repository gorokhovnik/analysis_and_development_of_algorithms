import numpy as np
from scipy.optimize import minimize, fmin_cg, least_squares, OptimizeResult
import matplotlib.pyplot as plt


def sgd(
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.001,
        mass=0.9,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs):
    x = x0
    velocity = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


np.random.seed(16777216)

alpha, beta = np.random.random(), np.random.random()

x = [k / 100 for k in range(101)]
y = [alpha * x_ + beta + np.random.normal() for x_ in x]


def f1(a):
    return sum([(a[0] * x_ + a[1] - y_) ** 2 for x_, y_ in zip(x, y)])


def ff1(a):
    return [sum([2 * (a[0] * x_ + a[1] - y_) * x_ for x_, y_ in zip(x, y)]),
            sum([2 * (a[0] * x_ + a[1] - y_) for x_, y_ in zip(x, y)])]


def fff1(a):
    return [[sum([2 * x_ * x_ for x_, y_ in zip(x, y)]), sum([2 * x_ for x_, y_ in zip(x, y)])],
            [sum([2 * x_ for x_, y_ in zip(x, y)]), sum([2 for x_, y_ in zip(x, y)])]]


def f2(a):
    return sum([(a[0] / (1 + a[1] * x_) - y_) ** 2 for x_, y_ in zip(x, y)])


def ff2(a):
    return [sum([(-2 * (a[1] * x_ * y_ + y_ - a[0])) / ((1 + a[1] * x_) ** 2) for x_, y_ in zip(x, y)]),
            sum([(2 * a[0] * x_ * (a[1] * x_ * y_ + y_ - a[0])) / ((1 + a[1] * x_) ** 3) for x_, y_ in zip(x, y)])]


def fff2(a):
    dxy = sum([(2 * x_ * (a[1] * x_ * y_ + y_ - 2 * a[0])) / ((1 + a[1] * x_) ** 3) for x_, y_ in zip(x, y)])
    return [[sum([2 / ((1 + a[1] * x_) ** 2) for x_, y_ in zip(x, y)]), dxy],
            [dxy, sum([(-4 * a[0] * x_ * x_ * (a[1] * x_ * y_ + y_ - 1.5 * a[0])) / ((1 + a[1] * x_) ** 4) for x_, y_ in
                       zip(x, y)])]]


def _f1(a):
    return [(a[0] * x_ + a[1] - y_) ** 2 for x_, y_ in zip(x, y)]


def _ff1(a):
    return [[2 * (a[0] * x_ + a[1] - y_) * x_, 2 * (a[0] * x_ + a[1] - y_)] for x_, y_ in zip(x, y)]


def _fff1(a):
    return [[[2 * x_ * x_ for x_, y_ in zip(x, y)], [2 * x_ for x_, y_ in zip(x, y)]],
            [[2 * x_ for x_, y_ in zip(x, y)], [2 for x_, y_ in zip(x, y)]]]


def _f2(a):
    return sum([(a[0] / (1 + a[1] * x_) - y_) ** 2 for x_, y_ in zip(x, y)])


def _ff2(a):
    return [sum([(-2 * (a[1] * x_ * y_ + y_ - a[0])) / ((1 + a[1] * x_) ** 2) for x_, y_ in zip(x, y)]),
            sum([(2 * a[0] * x_ * (a[1] * x_ * y_ + y_ - a[0])) / ((1 + a[1] * x_) ** 3) for x_, y_ in zip(x, y)])]


def _fff2(a):
    dxy = sum([(2 * x_ * (a[1] * x_ * y_ + y_ - 2 * a[0])) / ((1 + a[1] * x_) ** 3) for x_, y_ in zip(x, y)])
    return [[sum([2 / ((1 + a[1] * x_) ** 2) for x_, y_ in zip(x, y)]), dxy],
            [dxy, sum([(-4 * a[0] * x_ * x_ * (a[1] * x_ * y_ + y_ - 1.5 * a[0])) / ((1 + a[1] * x_) ** 4) for x_, y_ in
                       zip(x, y)])]]


# res1_cg = fmin_cg(f1, [0, 0], ff1, gtol=0.001)
# print(res1_cg)
# res2_cg = fmin_cg(f2, [0, 0], ff2, gtol=0.001)
# print(res2_cg)

# res1_n = minimize(f1, [0, 0], method='Newton-CG', jac=ff1, hess=fff1, options={'xtol': 0.001, 'disp': True})
# print(res1_n)
res1_n = minimize(f1, [0, 0], method=sgd, jac=ff1, options={'xtol': 0.001, 'disp': True})
print(res1_n)
# res2_n = minimize(f2, [0, 0], method='Newton-CG', jac=ff2, hess=fff2, options={'xtol': 0.001, 'disp': True})
# print(res2_n)

# res1_lm = least_squares(_f1, [0, 0], method='lm', gtol=0.001)
# print(res1_lm)
# res2_lm = least_squares(f2, [0, 0], method='lm', jac=ff2, gtol=0.001)
# print(res2_lm)
