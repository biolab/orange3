import warnings

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import scipy.sparse as sp

from Orange import classification
import Orange.data


# helper functions
def append_ones(X):
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cost_grad(theta, X, y, lambda_):
    sx = sigmoid(X.dot(theta))

    j = -np.log(np.where(y, sx, 1 - sx)).sum()
    j += lambda_ * theta.dot(theta) / 2.0
    j /= X.shape[0]

    grad = X.T.dot(sx - y)
    grad += lambda_ * theta
    grad /= X.shape[0]

    return j, grad


def fit(X, y, lambda_):
    theta = np.zeros(X.shape[1])
    theta, _, ret = fmin_l_bfgs_b(cost_grad, theta, args=(X, y, lambda_))
    if ret['warnflag'] != 0:
        warnings.warn('L-BFGS failed to converge')
    return theta


def predict(X, theta):
    return sigmoid(X.dot(theta))


# how it should look like
class LogisticRegressionSimple:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.theta = fit(X, y, self.lambda_)

    def predict(self, X):
        return predict(X, self.theta)


# how it would look like in Orange4.0
class LogisticRegressionLearner(classification.Learner):
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_

    def __call__(self, data):
        theta = fit(data.X, data.Y.ravel(), self.lambda_)
        return LogisticRegressionClassifier(theta)


class LogisticRegressionClassifier(classification.Classifier):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, X):
        return predict(X.X, self.theta)


if __name__ == '__main__':
    iris = Orange.data.Table('../doc/datasets/iris')
    iris.Y[iris.Y != 0.0] = 1.0

    m = LogisticRegressionSimple(lambda_=2)
    m.fit(iris.X, iris.Y.ravel())
    p_simple = m.predict(iris.X)

    c = LogisticRegressionLearner(lambda_=2)
    l = c(iris)
    p_orange = l(iris)

    assert np.allclose(p_simple, p_orange)
