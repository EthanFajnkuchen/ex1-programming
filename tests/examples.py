import numpy as np
from math import e


def Qi(x, calc_hessian):
    Q = np.eye(2)
    f = np.dot(x, Q.dot(x))
    g = 2 * Q.dot(x)
    h = 2 * Q if calc_hessian else None
    return f, g, h


def Qii(x, calc_hessian):
    Q = np.diag([1, 100])
    f = np.dot(x, Q.dot(x))
    g = 2 * Q.dot(x)
    h = 2 * Q if calc_hessian else None
    return f, g, h


def Qiii(x, calc_hessian):
    W = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q = np.diag([100, 1])
    Q = W.T @ Q @ W
    f = np.dot(x, Q.dot(x))
    g = 2 * Q.dot(x)
    h = 2 * Q if calc_hessian else None
    return f, g, h


def rosenbrock(x, calc_hessian):
    f = 100 * ((x[1] - x[0] ** 2) ** 2) + ((1 - x[0]) ** 2)
    g = np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
    h = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]]) if calc_hessian else None
    return f, g, h


def linear(x, calc_hessian):
    a = np.array([5, 2])
    f = a.T @ x
    g = a
    h = 0 if calc_hessian else None
    return f, g, h


def exp(x, calc_hessian):
    exp1, exp2, exp3 = e ** (x[0] + 3 * x[1] - 0.1), e ** (x[0] - 3 * x[1] - 0.1), e ** (-x[0] - 0.1)
    f = exp1 + exp2 + exp3
    g = np.array([2 * e ** x[0] - e ** -x[0], 3 * e ** (3 * x[1]) - 3 * e ** (-3 * x[1])])
    h = np.array([[e * e ** x[0] + e ** -x[0], 0], [0, 9 * e ** (3 * x[1]) + 9 * e ** (-3 * x[1])]]) if calc_hessian else None
    return f, g, h
