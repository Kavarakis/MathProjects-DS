import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from helper_f import bernoulli, sigmoid, normalPDF, ll, ll_grad


# dp(x) / dx = -p(x) * S^(-1) * (x - m)
def HM(n_steps, step_size, func, grad, current_q, seed):
    # np.random.seed(seed)
    q = current_q
    p = np.random.normal(size=len(q), loc=0, scale=1)
    current_p = p
    traj = []
    traj.append([p, q, func(q) + np.sum(p**2) / 2])
    p = p - step_size * grad(q) / 2
    for i in range(n_steps):
        q = q + step_size * p
        if (i < n_steps - 1):
            p = p - step_size * grad(q)

        traj.append([p, q, func(q) + np.sum(p**2) / 2])
    p = p - step_size * grad(q) / 2
    p = -1.0 * p
    current_U = func(current_q)
    current_K = np.sum(current_p**2) / 2
    proposed_U = func(q)
    proposed_K = np.sum(p**2) / 2
    alpha = np.exp(current_U - proposed_U + current_K - proposed_K)

    if (np.random.rand() < alpha):
        return q, traj
    else:
        return current_q, traj


def hamiltonian_mc(n, n_steps, step_size, func, grad, seed):
    samples = []
    traj_data = []
    init_q = np.array([0, 0])
    for i in range(n):
        # q, traj = HM(25, 0.4, biv_normal_f, biv_norm_grad, init_q)
        q, traj = HM(n_steps, step_size, func, grad, init_q, seed)
        samples.append([*q, 0])
        traj_data.append(traj)
        init_q = q
    return samples


def hm_logreg(x, y, n,n_steps,step_size, seed=0):
    np.random.seed(seed)
    params_n = x.shape[1]
    b = np.random.rand(params_n)

    def func(b):
        return -1.0 * ll(theta=b, x=x, y=y)

    def grad(b):
        return -1.0 * ll_grad(theta=b, x=x, y=y)

    samples = []
    traj_data = []
    init_q = normalPDF(b, 0, 1)
    n_steps = n_steps
    step_size = step_size
    for i in range(n):
        # q, traj = HM(25, 0.4, biv_normal_f, biv_norm_grad, init_q)
        q, traj = HM(n_steps, step_size, func, grad, init_q, seed)
        samples.append([*q, 0])
        traj_data.append(traj)
        init_q = q
    return samples


def generate_chains(alg, args):
    df = pd.DataFrame(columns=['x0', 'x1', 'isRejected'])
    prev_i = 0
    dfs = pd.DataFrame()
    for i in range(5):
        args.append(10 * int(i * np.random.rand()))
        t = pd.DataFrame(alg(*args), columns=['x0', 'x1', 'isRejected'])
        args.pop()
        t['chain_no'] = i + 1
        dfs = pd.concat([dfs, t])
    df = pd.concat([df, dfs], ignore_index=True)
    df['isRejected'] = df['isRejected'].astype(bool)
    df['chain_no'] = df['chain_no'].astype(int)
    return df


if __name__ == '__main__':

    logreg_data = pd.read_csv('datset.csv')
    true_val = [
        2.00, -0.84, -0.64, 0.72, -0.10, -0.85, -0.97, 0.23, -0.68, -0.42, 0.47
    ]
    cols = ['X' + str(i + 1) for i in range(11)]
    cols = cols[:2]
    x = logreg_data[cols].values
    y = logreg_data['y'].values

    hm_chains_logreg = np.array(hm_logreg(x, y, n=1_000))
    print(np.mean(hm_chains_logreg[:, 0]), np.mean(hm_chains_logreg[:, 1]))
    plt.plot(np.arange(1000), hm_chains_logreg[:, 0])
    plt.plot(np.arange(1000), hm_chains_logreg[:, 1])
    plt.show()
    n = 1000

    # samples = hamiltonian_mc(100, 15, 0.4, biv_normal_f, biv_norm_grad)
    # data = pd.DataFrame(samples, columns=['q0', 'q1'])
    # x = np.linspace(-20, 20, 100)
    # y = np.linspace(-20, 20, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = mult_normal_f(X, Y)
    # plt.contour(X, Y, Z)
    # data_dist = pd.DataFrame(stats.multivariate_normal(mean=np.zeros(
    #     cov.shape[0]),
    #                                                    cov=cov).rvs(1000),
    #                          columns=['x0', 'x1'])
    # sns.kdeplot(data=data_dist, x='x0', y='x1')
    # sns.scatterplot(data=data, x='q0', y='q1', color='green', alpha=0.8)
    # plt.show()
