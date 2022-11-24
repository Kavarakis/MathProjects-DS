import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def bernoulli(p, y):
    return (p**y) * ((1 - p)**(1 - y))


def ll(theta, x, y):

    s = np.dot(x, theta)
    _ll = np.sum(y * s - np.log(1 + np.exp(s)))
    return _ll


def ll_grad(theta, x, y):
    s = np.dot(x, theta)
    pred = sigmoid(s)
    _err = y - pred
    grad = x.T.dot(_err)

    return grad


def normalPDF(x, mu, sigma):
    num = np.exp(-1 / 2 * ((x - mu) / sigma)**2)
    den = np.sqrt(2 * np.pi) * sigma
    return num / den


def log_banana_f(x):
    B = 0.05
    return -(-(x[0]**2) / 200 - 0.5 * (x[1] + B * x[0]**2 - 100 * B)**2)


def banana_f_grad(x):
    #log
    B = 0.05
    g1 = -(x[0]) / 100. - 1.0 * (2 * B * x[0]) * (x[1] + B * x[0]**2 - 100 * B)
    g2 = -1.0 * (x[1] + B * x[0]**2 - 100 * B)
    return -1.0 * np.array([g1, g2])


cov = np.array([[1, 0.7], [0.7, 1]])


def biv_normal_f(x):
    return np.log(2 * np.pi) + 0.5 * np.sum(x.dot(x.T))


def biv_norm_grad(x):

    return x


def banana_f(x):
    return np.exp(-log_banana_f(x))
