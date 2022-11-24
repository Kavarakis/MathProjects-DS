import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


def generate_chains(alg, args, n_params=2):
    cols = ['x' + i for i in range(n_params)]
    cols.append('isRejected')
    df = pd.DataFrame(columns=cols)
    prev_i = 0
    dfs = pd.DataFrame()
    for i in range(5):
        args.append(10 * int(i * np.random.rand()))
        t = pd.DataFrame(alg(*args), columns=cols)
        args.pop()
        t['chain_no'] = i + 1
        dfs = pd.concat([dfs, t])
    df = pd.concat([df, dfs], ignore_index=True)
    df['isRejected'] = df['isRejected'].astype(bool)
    df['chain_no'] = df['chain_no'].astype(int)
    return df


def p(x):
    return stats.norm.pdf(x, loc=30, scale=10) + stats.norm.pdf(
        x, loc=80, scale=20)


def q(x):
    return stats.norm.pdf(x, loc=50, scale=30)


x = np.arange(-50, 151)
k = max(p(x) / q(x))


def rejection_sampling(p, q, cov, iter=1000, seed=0):
    q_pdf = lambda x, y: q.pdf(np.array([x, y]).T)
    samples = np.empty(shape=(iter, 3))
    sd = np.sqrt(cov[0, 0])
    x = np.arange(-3 * sd, 3 * sd, 0.001)

    k = max(p(x, x) / q_pdf(x, x))

    for i in range(iter):
        z = q.rvs()
        u = np.random.uniform(0, k * q_pdf(*z))

        if u <= p(*z):
            samples[i] = (np.hstack([z, 0]))
        else:
            samples[i] = (np.hstack([z, 1]))

    return samples


def rejection_sampling_ban(p, cov, iter=1000, seed=0):
    np.random.seed(seed)

    def q_pdf(x):
        B = 0.05
        dist = stats.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
        p2 = -1.0 * x[1] - B * x[0]**2 + 100 * B
        p1 = x[0]
        return dist.pdf(np.dstack((p1, p2)))

    def q_rvs():
        p = stats.multivariate_normal(mean=np.zeros(cov.shape[0]),
                                      cov=cov).rvs()
        p1 = p[0]
        p2 = -1.0 * (p[1] + 0.05 * p[0]**2 - 100 * 0.05)
        return np.array([p1, p2])

    samples = np.empty(shape=(iter, 3))
    sd1 = np.sqrt(cov[0, 0])
    sd2 = np.sqrt(cov[-1, -1])

    X = np.linspace(-3 * sd1, 3 * sd1, 100)
    Y = np.linspace(-3 * sd2, 3 * sd2, 100)
    x, y = np.meshgrid(X, Y)
    k = np.max(p(x, y) / q_pdf(np.array([x, y])))

    for i in range(iter):
        z = q_rvs()
        u = np.random.uniform(0, k * q_pdf(z))

        if u <= p(*z):
            samples[i] = (np.hstack([z, 0]))
        else:
            samples[i] = (np.hstack([z, 1]))

    return samples