import numpy as np
from scipy.optimize import minimize, fmin_cg, least_squares
import matplotlib.pyplot as plt
from algo_lib import gd


np.random.seed(16777216)

alpha, beta = np.random.random(), np.random.random()

x = np.array([k / 100 for k in range(101)])
y = np.array([alpha * x_ + beta + np.random.normal() for x_ in x])


def f1(a):
    return sum([(a[0] * x_ + a[1] - y_) ** 2 for x_, y_ in zip(x, y)])


def y1(a, x_):
    return a[0] * x_ + a[1]


def funs1(a):
    return y1(a, x) - y


def ff1(a):
    return [sum([2 * (a[0] * x_ + a[1] - y_) * x_ for x_, y_ in zip(x, y)]),
            sum([2 * (a[0] * x_ + a[1] - y_) for x_, y_ in zip(x, y)])]


def fff1(a):
    return [[sum([2 * x_ * x_ for x_, y_ in zip(x, y)]), sum([2 * x_ for x_, y_ in zip(x, y)])],
            [sum([2 * x_ for x_, y_ in zip(x, y)]), sum([2 for x_, y_ in zip(x, y)])]]


def f2(a):
    return sum([(a[0] / (1 + a[1] * x_) - y_) ** 2 for x_, y_ in zip(x, y)])


def y2(a, x_):
    return a[0] / (1 + a[1] * x_)


def funs2(a):
    return y2(a, x) - y


def ff2(a):
    return [sum([(-2 * (a[1] * x_ * y_ + y_ - a[0])) / ((1 + a[1] * x_) ** 2) for x_, y_ in zip(x, y)]),
            sum([(2 * a[0] * x_ * (a[1] * x_ * y_ + y_ - a[0])) / ((1 + a[1] * x_) ** 3) for x_, y_ in zip(x, y)])]


def fff2(a):
    dxy = sum([(2 * x_ * (a[1] * x_ * y_ + y_ - 2 * a[0])) / ((1 + a[1] * x_) ** 3) for x_, y_ in zip(x, y)])
    return [[sum([2 / ((1 + a[1] * x_) ** 2) for x_, y_ in zip(x, y)]), dxy],
            [dxy, sum([(-4 * a[0] * x_ * x_ * (a[1] * x_ * y_ + y_ - 1.5 * a[0])) / ((1 + a[1] * x_) ** 4) for x_, y_ in
                       zip(x, y)])]]


print('GD')
gd1 = gd(f1, [0, 0], ff1, tol=0.001)
print(gd1)
gd2 = gd(f2, [0, 0], ff2, tol=0.001)
print(gd2)

print('\n\nCGD')
cg1 = fmin_cg(f1, [0, 0], ff1, gtol=0.001)
print(cg1)
cg2 = fmin_cg(f2, [0, 0], ff2, gtol=0.001)
print(cg2)

print('\n\nNewton')
new1 = minimize(f1, [0, 0], method='Newton-CG', jac=ff1, hess=fff1, options={'xtol': 0.001, 'disp': True})
print(new1.x)
new2 = minimize(f2, [0, 0], method='Newton-CG', jac=ff2, hess=fff2, options={'xtol': 0.001, 'disp': True})
print(new2.x)

print('\n\nLM')
lm1 = least_squares(funs1, [0, 0], method='lm', gtol=0.001)
print((lm1.x.tolist(), lm1.nfev, f1(lm1.x)))
lm2 = least_squares(funs2, [0, 0], method='lm', gtol=0.001)
print((lm2.x.tolist(), lm2.nfev, f2(lm2.x)))

plt.title('GD // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([gd1[0][0] * x_ + gd1[0][1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('GD // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([gd2[0][0] / (1 + gd2[0][1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('CGD // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([cg1[0] * x_ + cg1[1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('CGD // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([cg2[0] / (1 + cg2[1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Newton // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([new1.x[0] * x_ + new1.x[1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('Newton // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([new2.x[0] / (1 + new2.x[1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('LMA // Linear', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([lm1.x[0] * x_ + lm1.x[1] for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()

plt.title('LMA // Rational', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot([lm2.x[0] / (1 + lm2.x[1] * x_) for x_ in x], color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()
