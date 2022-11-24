import numpy as np
import pandas as pd
from scipy import stats
from helper_f import sigmoid, bernoulli, ll, normalPDF


def prop(x, cov):
    return x + stats.multivariate_normal(mean=np.zeros(cov.shape[0]),
                                         cov=cov).rvs()


def metropolis_hastings(p, cov, proposal=prop, iter=1000, seed=0):
    np.random.seed(seed)

    x, y = 0., 0
    samples = np.zeros((iter, cov.shape[0] + 1))

    for i in range(iter):
        x_star, y_star = proposal(np.array([x, y]), cov)
        if np.random.rand() < p(np.array([x_star, y_star])) / p(
                np.array([x, y])):
            x, y = x_star, y_star
            samples[i] = np.array([x, y, 0])
        else:
            samples[i] = np.array([x, y, 1])

    return samples


def mh_logreg(x, y, cov, n=1_000, seed=0):
    np.random.seed(seed)
    y = y.reshape((len(y), 1))
    params_n = x.shape[1]
    b = np.random.rand(params_n)
    prior_b = normalPDF(b, 0, 1)
    z = x.dot(b.T)
    res = sigmoid(z)
    res = res.reshape(len(res), 1)
    samples = np.zeros((n, params_n + 1))
    log_like = np.sum(np.log(bernoulli(res, y)))
    x0 = log_like + np.sum(np.log(prior_b))
    for i in range(n):
        prop_b = b + np.random.normal(size=b.shape[0], loc=0, scale=cov[0, 0])
        pprop_b = normalPDF(prop_b, 0, cov[0, 0])
        z = x.dot(prop_b.T)
        res = sigmoid(z)
        res = res.reshape(len(res), 1)
        prop_log_like = np.sum(np.log(bernoulli(res, y)))
        x1 = prop_log_like + np.sum(np.log(pprop_b))
        alpha = np.exp(x1 - x0)
        if np.random.rand() < alpha:
            b = prop_b
            samples[i] = np.hstack([prop_b, 0])
        else:
            samples[i] = np.hstack([prop_b, 1])
        a = 2
    return samples


def logreg_mh(x, y, cov, n=1_000, seed=0):
    np.random.seed(seed)
    y = y.reshape((len(y), 1))

    b0, b1 = np.random.rand(), np.random.rand()

    prior_b0, prior_b1 = normalPDF(b0, 0, 1), normalPDF(b1, 0, 1)
    z = x.dot(np.array([[b0, b1]]).T)
    res = sigmoid(z)
    samples = np.zeros((n, cov.shape[0] + 1))
    log_like = np.sum(np.log(bernoulli(res, y)))
    x0 = log_like + np.log(prior_b0) + np.log(prior_b1)
    p = 0

    for i in range(n):
        prop_b0, prop_b1 = b0 + np.random.normal(), b1 + np.random.normal()
        # b0 + np.random.uniform(-0.5, 0.5), b1 + np.random.uniform(-0.5, 0.5)
        pprop_b0, pprop_b1 = normalPDF(prop_b0, 0, 1), normalPDF(prop_b1, 0, 1)
        z = x.dot(np.array([[prop_b0, prop_b1]]).T)
        res = sigmoid(z)
        prop_log_like = np.sum(np.log(bernoulli(res, y)))
        x1 = prop_log_like + np.log(pprop_b0) + np.log(pprop_b1)

        alpha = np.exp(x1 - x0)
        if np.random.rand() < alpha:
            b0, b1 = prop_b0, prop_b1
            samples[i] = np.array([prop_b0, prop_b1, 0])
        else:
            samples[i] = np.array([b0, b1, 1])

    return samples


if __name__ == '__main__':
    logreg_data = pd.read_csv('datset.csv')
    true_val = [
        2.00, -0.84, -0.64, 0.72, -0.10, -0.85, -0.97, 0.23, -0.68, -0.42, 0.47
    ]
    cols = ['X' + str(i + 1) for i in range(11)]
    x = logreg_data[cols].values
    y = logreg_data['y'].values
    params_n = x.shape[1]
    b = np.random.rand(params_n)
    # cov = np.array([[1, 0.7], [0.7, 1]])
    # s = mh_logreg(x, y, cov, 10_000, seed=14)
    # cols.append('isRejected')
    # data = pd.DataFrame(s, columns=cols)
    # print(data['isRejected'].value_counts())
    # b = np.apply_along_axis(func1d=lambda x: np.mean(x), axis=0, arr=s[:, :-1])
    # res = sigmoid(x.dot(b))
    # print(b)
    # print(np.mean([bool(i > 0.5) for i in res] == y))
