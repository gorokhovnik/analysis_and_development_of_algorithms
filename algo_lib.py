import numpy as np
from functools import partial


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


def gd(f,
       x0,
       ff,
       tol=0.001):
    prev_z = np.inf
    z = f(x0)
    n = 1
    while prev_z - z > tol:
        d = ff(x0)
        x0[0], x0[1] = x0[0] - d[0] * tol * 7, x0[1] - d[1] * tol * 7
        prev_z = z
        z = f(x0)
        n += 1
    return x0, n, z


def recursive_find_match(i, j, pattern, pattern_track):
    if pattern[i] == pattern[j]:
        pattern_track.append(i + 1)
        return {"append": pattern_track, "i": i + 1, "j": j + 1}
    elif pattern[i] != pattern[j] and i == 0:
        pattern_track.append(i)
        return {"append": pattern_track, "i": i, "j": j + 1}

    else:
        i = pattern_track[i - 1]
        return recursive_find_match(i, j, pattern, pattern_track)


def kmp(str_, pattern):
    len_str = len(str_)
    len_pattern = len(pattern)
    pattern_track = []

    if len_pattern == 0:
        return
    elif len_pattern == 1:
        pattern_track = [0]
    else:
        pattern_track = [0]
        i = 0
        j = 1

        while j < len_pattern:
            data = recursive_find_match(i, j, pattern, pattern_track)

            i = data["i"]
            j = data["j"]
            pattern_track = data["append"]

    index_str = 0
    index_pattern = 0
    match_from = -1

    while index_str < len_str:
        if index_pattern == len_pattern:
            break
        if str_[index_str] == pattern[index_pattern]:
            if index_pattern == 0:
                match_from = index_str

            index_pattern += 1
            index_str += 1
        else:
            if index_pattern == 0:
                index_str += 1
            else:
                index_pattern = pattern_track[index_pattern - 1]
                match_from = index_str - index_pattern
