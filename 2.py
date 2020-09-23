import numpy as np
from functools import partial
import matplotlib.pyplot as plt

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


def exhaustive_search_1dim(f,
                           x_min=0,
                           x_max=1,
                           eps=0.001):
    curr_x = x_min
    best_x = x_min
    best_z = f(curr_x)
    n = 1
    i = 1

    while curr_x < x_max:
        n += 1
        i += 1
        curr_x += eps
        z = f(curr_x)
        if z < best_z:
            best_z = z
            best_x = curr_x

    return n, i, best_x, best_z


def dichotomy(f,
              x_min=0,
              x_max=1,
              eps=0.001):
    n = 0
    i = 0
    a = x_min
    b = x_max

    while b - a > eps:
        c = (a + b) / 2
        if f(c + eps) > f(c - eps):
            b = c
        else:
            a = c
        n += 2
        i += 1

    return n, i, (a + b) / 2, f((a + b) / 2)


def golden_section(f,
                   x_min=0,
                   x_max=1,
                   eps=0.001):
    n = 2
    i = 1
    a = x_min
    b = x_max
    c1 = (a + b) * 0.382
    c2 = (a + b) * 0.618

    f_c1 = f(c1)
    f_c2 = f(c2)

    while c2 - c1 > eps:
        if f_c2 > f_c1:
            b = c2
            c2 = c1
            f_c2 = f_c1
            c1 = (b - a) * 0.382 + a
            f_c1 = f(c1)
        else:
            a = c1
            c1 = c2
            f_c1 = f_c2
            c2 = (b - a) * 0.618 + a
            f_c2 = f(c2)

        n += 1
        i += 1

    return n, i, (c1 + c2) / 2, f((c1 + c2) / 2)


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


def exhaustive_search_2dim(f,
                           x_min=0,
                           x_max=1,
                           y_min=0,
                           y_max=1,
                           eps=0.001):
    n = 0
    i = 0
    curr_x = x_min
    curr_y = y_min
    best_xy = (x_min, y_min)
    best_z = np.inf

    while curr_x < x_max:
        while curr_y < y_max:
            z = f(curr_x, curr_y)
            n += 1
            i += 1
            if z < best_z:
                best_z = z
                best_xy = (curr_x, curr_y)
            curr_y += eps
        curr_y = y_min
        curr_x += eps

    return n, i, best_xy, best_z


def gauss(f,
          x_min=0,
          x_max=1,
          y_min=0,
          y_max=1,
          eps=0.001):
    n = 0
    i = 0
    curr_x = x_min
    curr_y = y_min
    best_z = np.inf

    while True:
        n_, i_, curr_x, z = exhaustive_search_1dim(partial(f, b=curr_y),
                                                   x_min=x_min,
                                                   x_max=x_max,
                                                   eps=eps)
        n += n_
        i += 1
        if best_z - z < eps:
            best_z = z
            break
        best_z = z
        n_, i_, curr_y, z = exhaustive_search_1dim(partial(f, curr_x),
                                                   x_min=y_min,
                                                   x_max=y_max,
                                                   eps=eps)
        n += n_
        i += 1
        if best_z - z < eps:
            best_z = z
            break
        best_z = z

    return n, i, (curr_x, curr_y), best_z


def nelder_mead(f,
                x_min=0,
                x_max=1,
                y_min=0,
                y_max=1,
                eps=0.001,
                alpha=1,
                beta=0.5,
                gamma=2):
    n = 3
    i = 1
    p1, p2, p3 = (x_min, y_min), (x_min, y_max), (x_max, y_min)
    f_p1, f_p2, f_p3 = f(p1[0], p1[1]), f(p2[0], p2[1]), f(p3[0], p3[1])

    if f_p1 > f_p2 > f_p3:
        f_ph, f_pg, f_pl = f_p1, f_p2, f_p3
        ph, pg, pl = p1, p2, p3
    elif f_p1 > f_p3 > f_p2:
        f_ph, f_pg, f_pl = f_p1, f_p3, f_p2
        ph, pg, pl = p1, p3, p2
    elif f_p2 > f_p1 > f_p3:
        f_ph, f_pg, f_pl = f_p2, f_p1, f_p3
        ph, pg, pl = p2, p1, p3
    elif f_p2 > f_p3 > f_p1:
        f_ph, f_pg, f_pl = f_p2, f_p3, f_p1
        ph, pg, pl = p2, p3, p1
    elif f_p3 > f_p1 > f_p2:
        f_ph, f_pg, f_pl = f_p3, f_p1, f_p2
        ph, pg, pl = p3, p1, p2
    else:
        f_ph, f_pg, f_pl = f_p3, f_p2, f_p1
        ph, pg, pl = p3, p2, p1

    while True:
        pc = ((pg[0] + pl[0]) / 2, (pg[1] + pl[1]) / 2)
        pr = ((pc[0] - ph[0]) * alpha + pc[0], (pc[1] - ph[1]) * alpha + pc[1])
        f_pr = f(pr[0], pr[1])
        n += 1
        if f_pr < f_pl:
            pe = ((pr[0] - pc[0]) * gamma + pr[0], (pr[1] - pc[1]) * gamma + pr[1])
            f_pe = f(pe[0], pe[1])
            n += 1
            if f_pe < f_pr:
                ph, pg, pl = pg, pl, pe
                f_ph, f_pg, f_pl = f_pg, f_pl, f_pe
            else:
                ph, pg, pl = pg, pl, pr
                f_ph, f_pg, f_pl = f_pg, f_pl, f_pr
        elif f_pl < f_pr < f_pg:
            ph, pg, pl = pg, pl, pr
            f_ph, f_pg, f_pl = f_pg, f_pl, f_pr
        elif f_pg < f_pr:
            if f_pr < f_ph:
                ph, f_ph = pr, f_pr
            ps = ((ph[0] - pc[0]) * beta + pc[0], (ph[1] - pc[1]) * beta + pc[1])
            f_ps = f(ps[0], ps[1])
            n += 1
            if f_ps < f_ph:
                ph, f_ph = ps, f_ps
            else:
                ph = ((ph[0] + pl[0]) / 2, (ph[1] + pl[1]) / 2)
                pg = ((pg[0] + pl[0]) / 2, (pg[1] + pl[1]) / 2)
                f_ph, f_pg = f(ph[0], ph[1]), f(pg[0], pg[1])
                n += 1
                if f_ph > f_pl > f_pg:
                    pg, pl = pl, pg
                    f_pg, f_pl = f_pl, f_pg
                elif f_pg > f_ph > f_pl:
                    ph, pg = pg, ph
                    f_ph, f_pg = f_pg, f_ph
                elif f_pg > f_pl > f_ph:
                    ph, pg, pl = pg, pl, ph
                    f_ph, f_pg, f_pl = f_pg, f_pl, f_ph
                elif f_pl > f_ph > f_pg:
                    ph, pg, pl = pl, ph, pg
                    f_ph, f_pg, f_pl = f_pl, f_ph, f_pg
                elif f_pl > f_pg > f_ph:
                    ph, pl = pl, ph
                    f_ph, f_pl = f_pl, f_ph

        i += 1

        if max([abs(ph[0] - pg[0]), abs(ph[0] - pl[0]), abs(pl[0] - pg[0]),
                abs(ph[1] - pg[1]), abs(ph[1] - pl[1]), abs(pl[1] - pg[1])]) < eps:
            break

    return n, i, pl, f_pl


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
