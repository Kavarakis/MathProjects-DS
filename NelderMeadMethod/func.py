import numpy as np
from subprocess import getstatusoutput


def f1(x):
    return (x[0] - x[2])**2 + (2 * x[1] + x[2])**2 + (4 * x[0] - 2 * x[1] +
                                                      x[2])**2 + x[0] + x[1]


def f1prim(x):
    return np.array([(34 * x[0] - 16 * x[1] + 6 * x[2] + 1),
                     (-16 * x[0] + 16 * x[1] + 1), 6 * (x[0] + x[2])])


def f1primprim(x):
    return np.array([34, 16, 6])


def func2(x):
    return (x[0] - 1)**2 + (
        x[1] - 1)**2 + 100 * (x[1] - x[0]**2)**2 + 100 * (x[2] - x[1]**2)**2


def f2prim(x):
    return np.array([
        (400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2),
        (-200 * x[0]**2 + 400 * x[1]**3 + x[1] * (202 - 400 * x[2]) - 2),
        200 * (x[2] - x[1]**2)
    ])


def f2primprim(x):
    return np.array([
        1200 * x[0]**2 - 400 * x[1] + 2, 1200 * x[1]**2 + 202 - 400 * x[2], 200
    ])


def f3(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + (
        2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * (x[1]**3))**2


def f3prim(x):
    return np.array([
        (2 * x[0] * (x[1]**6 + x[1]**4 - x[1]**2 - 2 * x[1] + 3) +
         5.25 * x[1]**3 + 4.5 * x[1]**2 + 3 * x[1] - 12.75),
        (6 * x[0] *
         (x[0] *
          (x[1]**5 + 0.666667 * x[1]**3 - x[1]**2 - 0.333333 * x[1] - 0.333333)
          + 2.625 * x[1]**2 + 1.5 * x[1] + 0.5))
    ])


def f3primprim(x):
    return np.array([
        2 * (x[1]**6 + x[1]**4 - 2 * x[1]**2 - 2 * x[1] + 3),
        (30 * (x[1]**4) * x[0]**2 + 12 * (x[0]**2) * (x[1]**2) - 12 *
         (x[0]**2) * x[1] - 1.99 * x[0]**2 + 31.5 * x[1] * x[0] + 9 * x[0])
    ])


def black_f(x, i=1):
    st_id = '63190409'
    s = f"./hw4_nix {st_id} {i} {x[0]} {x[1]} {x[2]}"
    res = float(getstatusoutput(s)[1])
    return res