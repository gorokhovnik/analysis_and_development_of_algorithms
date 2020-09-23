import numpy as np
from scipy.optimize import minimize, fmin_cg, least_squares, leastsq
import matplotlib.pyplot as plt

np.random.seed(16777216)


def F(x_):
    return 1 / (x_ ** 2 - 3 * x_ + 2)


x = np.array([3 * k / 1000 for k in range(1001)])
y = np.array([-100 + np.random.normal() if F(x_) < -100 else
              100 + np.random.normal() if F(x_) > 100 else
              F(x_) + np.random.normal() for x_ in x])

plt.plot(x, y)
plt.show()
