import numpy as np
from scipy import stats
import pandas as pd
from mh import metropolis_hastings, mh_logreg
from hm import hamiltonian_mc, hm_logreg
from helper_f import biv_norm_grad, biv_normal_f, banana_f, log_banana_f, banana_f_grad, sigmoid, bernoulli
from r_sam import rejection_sampling, rejection_sampling_ban
from time import time
import matplotlib.pyplot as plt
import seaborn as sns


def prop_mh(x, cov):
    return x + stats.multivariate_normal(mean=np.zeros(cov.shape[0]),
                                         cov=cov).rvs()


def proposalBan(x, cov):

    p = stats.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov).rvs()
    x[0] += p[0]
    x[1] += -1.0 * (p[1] + 0.05 * p[0]**2 - 100 * 0.05)
    return x


def generate_chains(alg, args, n_params=2):
    cols = ['x' + str(i) for i in range(n_params)]
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


if __name__ == '__main__':
    times = {
        "mh_biv": 0,
        "hm_biv": 0,
        "mh_ban": 0,
        "hm_ban": 0,
        "rej_biv": 0,
        "rej_ban": 0,
        "mh_lr_2": 0,
        "hm_lr_2": 0,
        "rej_lr": 0,
        "mh_lr": 0,
        "hm_lr": 0
    }

    def mh_biv():
        bn_dist = stats.multivariate_normal(mean=np.zeros(2),
                                            cov=np.array([[1, 0], [0, 1]]))

        def target_dist(x, ):
            return bn_dist.pdf(x)

        cov = np.array([[2, 0], [0, 2]])

        start = time()
        mh_chains_biv = generate_chains(metropolis_hastings,
                                        [target_dist, cov, prop_mh, 1_000])
        end = time()
        times['mh_biv'] = end - start
        print('Finished with mh_chain_biv...')
        return mh_chains_biv

    # 15 0.4

    def hm_biv():

        start = time()
        hm_chains_biv = generate_chains(
            hamiltonian_mc, [1000, 5, 0.3, biv_normal_f, biv_norm_grad])
        end = time()
        times['hm_biv'] = end - start
        print('Finished with hm_chain_biv...')
        return hm_chains_biv

    def mh_ban():
        start = time()
        _cov = np.array([[75, 0], [0, 2]])
        mh_chains_ban = generate_chains(metropolis_hastings,
                                        [banana_f, _cov, proposalBan, 1_000])
        end = time()
        times['mh_ban'] = end - start

        print('Finished with mh_chain_ban...')
        return mh_chains_ban

    def hm_ban():
        start = time()

        hm_chains_ban = generate_chains(
            hamiltonian_mc, [1000, 30, 0.8, log_banana_f, banana_f_grad])
        end = time()
        times['hm_ban'] = end - start

        print('Finished with hm_chain_ban...')
        return hm_chains_ban

    def rej_biv():
        cov1 = np.array([[1, 0.2], [0.2, 1]])
        dist_p = stats.multivariate_normal(mean=np.zeros(cov1.shape[0]),
                                           cov=cov1)
        cov2 = np.array([[1.5, 0.5], [0.5, 1.5]])
        dist_q = stats.multivariate_normal(mean=np.zeros(cov2.shape[0]),
                                           cov=cov2)
        start = time()
        rej_chains_biv = generate_chains(
            rejection_sampling,
            [lambda x, y: dist_p.pdf(np.array([x, y]).T), dist_q, cov1, 1000])
        end = time()
        times['rej_biv'] = end - start

        print('Finished with rej_chain_biv...')
        return rej_chains_biv

    def rej_ban():
        cov = np.array([[100, 0], [0, 2]])
        start = time()
        rej_chains_ban = generate_chains(
            rejection_sampling_ban,
            [lambda x, y: banana_f(np.array([x, y])), cov, 1000])
        print('Finished with rej_chain_ban...')
        end = time()
        times['rej_ban'] = end - start
        return rej_chains_ban

    logreg_data = pd.read_csv('datset.csv')
    true_val = [
        2.00, -0.84, -0.64, 0.72, -0.10, -0.85, -0.97, 0.23, -0.68, -0.42, 0.47
    ]
    cols = ['X' + str(i + 1) for i in range(11)]

    def mh_lr2():
        x_val = logreg_data[cols[:2]].values
        y_val = logreg_data['y'].values
        cov = np.array([[4, 0.7], [0.7, 4]])
        start = time()
        mh_lr_s = generate_chains(mh_logreg, [x_val, y_val, cov, 1_000])
        end = time()
        times['mh_lr_2'] = end - start
        print('Finished with mh_logreg_2...')
        return mh_lr_s

    def hm_lr2():
        x_val = logreg_data[cols[:2]].values
        y_val = logreg_data['y'].values
        start = time()
        hm_lr_s = generate_chains(hm_logreg, [x_val, y_val, 1_000, 25, 0.1])
        end = time()
        times['hm_lr_2'] = end - start

        print('Finished with hm_logreg_2...')
        return hm_lr_s

    def rej_lr():
        x_val = logreg_data[cols[:2]].values
        y_val = logreg_data['y'].values

        def dist_p(x0, x1):

            b = np.array([x0, x1])
            b = b.T
            r = []
            if (len(b.shape) == 1):
                res = sigmoid(x_val.dot(b.T))
                return np.prod(bernoulli(res, y_val))
            for i in range(b.shape[0]):
                res = sigmoid(x_val.dot(b[i, :].T))
                r.append(np.prod(bernoulli(res, y_val)))
            return r

        cov2 = np.array([[0.2, 0.0005], [0.0005, 0.2]])
        dist_q = stats.multivariate_normal(mean=np.zeros(cov2.shape[0]),
                                           cov=cov2)
        start = time()
        rej_chains_lr = generate_chains(rejection_sampling,
                                        [dist_p, dist_q, cov2, 1000])
        end = time()
        times['rej_lr'] = end - start

        print('Finished with rej_logreg_2...')
        return rej_chains_lr

    def mh_lr_full():

        x_val = logreg_data[cols].values
        y_val = logreg_data['y'].values
        cov = np.array([[3.65, 0.7], [0.7, 1]])

        start = time()

        mh_lr_s_full = generate_chains(mh_logreg, [x_val, y_val, cov, 1_000],
                                       x_val.shape[1])
        end = time()
        times['mh_lr'] = end - start

        print('Finished with mh_logreg_full...')
        return mh_lr_s_full

    def hm_lr_full():
        x_val = logreg_data[cols].values
        y_val = logreg_data['y'].values
        start = time()
        hm_lr_s_full = generate_chains(hm_logreg,
                                       [x_val, y_val, 1_000, 15, 0.1],
                                       x_val.shape[1])
        end = time()
        times['hm_lr'] = end - start
        print('Finished with hm_logreg_full...')
        return hm_lr_s_full

    mh_biv().to_csv('mh_chains_biv.csv', index=False)
    # hm_biv().to_csv('hm_chains_biv.csv', index=False)
    # rej_biv().to_csv('rej_chains_biv.csv', index=False)
    # mh_ban().to_csv('mh_chains_ban.csv', index=False)
    # hm_ban().to_csv('hm_chains_ban.csv', index=False)
    # rej_ban().to_csv('rej_chains_ban.csv', index=False)
    # hm_lr2().to_csv('hm_lr_s.csv', index=False)
    # mh_lr2().to_csv('mh_lr_s.csv', index=False)
    # rej_lr().to_csv('rej_chains_lr.csv', index=False)
    # hm_lr_full().to_csv('hm_lr_s_full.csv', index=False)
    # mh_lr_full().to_csv('mh_lr_s_full.csv', index=False)
    # pd.json_normalize(times).to_csv('times.csv', index=False)

    exit()