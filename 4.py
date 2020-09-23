import numpy as np
from scipy.optimize import minimize, least_squares, dual_annealing, differential_evolution
import matplotlib.pyplot as plt

np.random.seed(16777216)


def f_(x_):
    return 1 / (x_ ** 2 - 3 * x_ + 2)


x = np.array([3 * k / 1000 for k in range(1001)])
y = np.array([-100 + np.random.normal() if f_(x_) < -100 else
              100 + np.random.normal() if f_(x_) > 100 else
              f_(x_) + np.random.normal() for x_ in x])


def f(a):
    return sum([((a[0] * x_ + a[1]) / (x_ * x_ + a[2] * x_ + a[3]) - y_) ** 2 for x_, y_ in zip(x, y)])


def fu(a, x_):
    return (a[0] * x_ + a[1]) / (x * x + a[2] * x + a[3])


def fun(a):
    return fu(a, x) - y


print('NM')
nm = minimize(f, [1, 1, 1, 1], method='nelder-mead', options={'xtol': 0.001, 'disp': True})
print(nm.x)

print('\n\nLM')
lm = least_squares(fun, [1, 1, 1, 1], method='lm', gtol=0.001)
print((lm.x.tolist(), lm.nfev, f(lm.x)))

print('\n\nSA')
sa = dual_annealing(f, list(zip([-3] * 4, [3] * 4)), seed=16777216)
print((sa.x.tolist(), sa.nfev, sa.nit, f(sa.x)))

print('\n\nDE')
de = differential_evolution(f, list(zip([-3] * 4, [3] * 4)), tol=0.001, seed=16777216)
print((de.x.tolist(), de.nfev, de.nit, f(de.x)))

plt.title('Compare', fontsize=20)
plt.plot(y, color='red', label='experimental')
plt.plot(fu(nm.x, x), color='magenta', linewidth=10.5, label='NM')
plt.plot(fu(lm.x, x), color='yellow', linewidth=7.5, label='LM')
plt.plot(fu(sa.x, x), color='orange', linewidth=4.5, label='SA')
plt.plot(fu(de.x, x), color='cyan', linewidth=1.5, label='DE')
plt.legend(loc='upper right')
plt.show()
