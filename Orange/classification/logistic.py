import logging

import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b

from Orange import classification

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LogisticRegressionLearner(classification.Fitter):
    def __init__(self, lambda_, normalize=True, **fmin_args):
        self.lambda_ = lambda_
        self.fmin_args = fmin_args
        self.normalize = normalize

    def cost_grad(self, theta, X, y):
        sx = np.clip(sigmoid(X.dot(theta)), 1e-15, 1 - 1e-15)

        cost = -np.log(np.where(y, sx, 1 - sx)).sum()
        cost += self.lambda_ * theta.dot(theta)
        cost /= X.shape[0]

        grad = X.T.dot(sx - y)
        grad += self.lambda_ * theta
        grad /= X.shape[0]

        return cost, grad

    def fit(self, X, Y, W):
        if list(np.unique(Y).astype(int)) != [0, 1]:
            raise ValueError('Logistic regression requires a binary class '
                'variable')

        if Y.shape[1] > 1:
            raise ValueError('Logistic regression does not support '
                'multi-label classification')

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if self.normalize:
            X = (X - self.mean) / self.std
        
        theta = np.zeros(X.shape[1])
        theta, cost, ret = fmin_l_bfgs_b(
            self.cost_grad, theta, args=(X, Y.ravel()), **self.fmin_args)
        if ret['warnflag'] != 0:
            logging.info('L-BFGS failed to converge')

        return LogisticRegressionClassifier(theta, self.normalize, self.mean, self.std)

class LogisticRegressionClassifier(classification.Model):
    def __init__(self, theta, normalize, mean, std):
        self.theta = theta
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
    def predict(self, X):
        if self.normalize:
            X = (X - self.mean) / self.std
        prob = sigmoid(X.dot(self.theta))
        return np.column_stack((1 - prob, prob))

if __name__ == '__main__':
    import Orange.data
    from sklearn.cross_validation import StratifiedKFold

    data = Orange.data.Table('../tests/iris')
    X, Y = data.X, data.Y
    Y[Y == 0.0] = 1.0
    Y[Y == 2.0] = 0.0

    m = LogisticRegressionLearner(lambda_=1.0, normalize=False)
    scores = []
    for tr_ind, te_ind in StratifiedKFold(Y.ravel()):
        s = np.mean(m(data[tr_ind])(data[te_ind]) == data[te_ind].Y.ravel())
        scores.append(s)
    print(np.mean(scores))
        
