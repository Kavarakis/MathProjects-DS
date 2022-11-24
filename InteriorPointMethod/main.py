from cProfile import label
from cmath import exp
from unittest import main
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, linprog


def f(x):
    return (x[0] - 5)**2 + (x[1] - 6)**2


def IPM(A, b, c):
    m, n = A.shape
    tol = np.exp(-25)
    i = 0
    _a = np.max(np.abs(A))
    _b = np.max(np.abs(b))
    _c = np.max(np.abs(c))
    x = np.ones(n + 2)
    e = np.ones(n)
    U = np.max([_a, _b, _c])
    W = (m * U)**m
    R = np.float(1 / W**2) * (1 / np.float((2 * n * (
        (m + 1) * U)**(3 * (m + 1))) + np.exp(-100)))

    M = (4.0 * n * U) / R
    Q = R / (n + 2)
    mi_f = (R * Q) / np.float(64 * (((m + 1) * U)**(m + 2)) * (n + 2)**2)
    d = b / W
    ro = d - A.dot(e)
    ro = ro.reshape((ro.shape[0], 1))
    A_prim = np.hstack([A, np.zeros(shape=(A.shape[0], 1)), ro])
    A_prim = np.vstack([A_prim, np.ones(n + 2)])
    b_prim = np.hstack([d, n + 2])
    # b_prim = b_prim.reshape((b_prim.shape[0], 1))
    mi = np.sqrt(4.0 * (np.float(M**2) + np.sum(c**2)))
    s = np.hstack([c + mi * e, mi, M + mi])
    y = np.hstack([np.zeros(m), -mi])
    delta = 1 / (8 * np.sqrt(n + 2))

    xs = []
    costx = []
    costy = []

    _mi = []
    _sigma_mi = []

    while (1):

        S = np.diag(s)
        X = np.diag(x)
        e = np.ones(x.shape[0])
        inv_S = np.linalg.inv(S)
        k1 = np.dot(A_prim, inv_S)
        k1 = np.dot(k1, np.dot(X, A_prim.T))
        k1 = np.linalg.inv(k1)
        k2 = b_prim - mi * np.dot(np.dot(A_prim, inv_S), e)
        k = np.dot(k1, k2)
        f = -np.dot(A_prim.T, k)
        h = np.dot(-np.dot(X, inv_S), f) + mi * np.dot(inv_S, e) - x

        # LR
        lr = 1
        k = lr * k
        h = lr * h
        f = lr * f
        x = x + h

        s = s + f
        y = y + k

        mi = (1 - delta) * mi
        # mi = np.sum(x * s)
        # mi = update_mi(x, s, mi)
        sigma_mi = np.sum((x * s / mi - 1)**2)
        costx.append(c.T.dot(x[:-2]))
        costy.append(d.T.dot(y[:-1]))

        xs.append(np.sum(x * s))
        _mi.append(mi)
        _sigma_mi.append(sigma_mi)

        i += 1
        if (mi < mi_f or np.sum(x * s) < tol):
            break
        print(f'Step ${i+1}-----\n')
    plt.loglog(np.arange(len(costx)), costx)
    plt.grid()
    plt.xlabel('log(Steps)')
    plt.ylabel('log(f)')
    plt.title('Cost function')
    plt.savefig('cost.png')
    plt.clf()

    # plt.plot(np.arange(len(xs)),
    #          np.log([np.exp(-20) if i < 0 else i for i in xs]))
    plt.loglog(np.arange(len(xs)), xs)
    plt.grid()
    plt.xlabel('log(Steps)')
    plt.title('Weak Duality hypothesis')
    plt.ylabel('$\log(x^Ts)$')
    plt.tight_layout()
    plt.savefig('xs.png')
    plt.clf()
    plt.plot(np.arange(len(_sigma_mi)), _sigma_mi)
    plt.grid()
    plt.xlabel('steps')
    plt.title('Iteration invariant - $\sigma$')
    plt.ylabel('$\sigma$')
    plt.tight_layout()

    plt.savefig('sigma.png')
    plt.clf()
    # s[np.where(s < Q / (4 * (n + 2)))] = 0
    # x[np.where(x < Q / (4 * (n + 2)))] = 0
    s[np.where(s < 8 * mi / Q)] = 0
    x[np.where(x < 8 * mi / Q)] = 0

    return 0


A = np.array([[18, 48, 5, 1, 5, 0, 0, 8], [2, 11, 3, 13, 3, 0, 15, 1],
              [0, 5, 3, 10, 3, 100, 30, 1],
              [77, 270, 60, 140, 61, 880, 330, 32]])
# A = np.vstack([A, -A])
# b = np.array([250, 50, 50, 2200])
b = np.array([370, 170, 90, 2400])
c = np.array([10, 22, 15, 45, 40, 20, 87, 21])
# A = np.hstack([A, np.zeros(A.shape)])
# A = np.vstack([A, np.eye(A.shape[1])])
# b = np.hstack([b, np.zeros(A.shape[1])])
# c = np.hstack([c, np.zeros(c.shape[0])])
# IPM(A, b, c, np.array([14, 3]))
# interior_point_method(A, b, c)
res = linprog(c=c, A_ub=A, b_ub=b)

IPM(A, b, c)
print('xxx')