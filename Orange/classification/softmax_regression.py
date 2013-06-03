import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b

from Orange import classification


class SoftmaxRegressionLearner(classification.Fitter):
    def __init__(self, lambda_, normalize=True, **fmin_args):
        self.lambda_ = lambda_
        self.fmin_args = fmin_args

    def cost_grad(self, Theta_flat, X, Y, y):
        Theta = Theta_flat.reshape((self.num_classes, X.shape[1]))

        P = np.exp(X.dot(Theta.T))
        P /= np.sum(P, axis=1)[:, None]

        cost = -np.sum(np.log(P[range(y.size), y]))
        cost += self.lambda_ * Theta_flat.dot(Theta_flat) / 2.0
        cost /= X.shape[0]

        grad = X.T.dot(P - Y).T
        grad += self.lambda_ * Theta
        grad /= X.shape[0]

        return cost, grad.ravel()

    def fit(self, X, y, W):
        if y.shape[1] > 1:
            raise ValueError('Softmax regression does not support '
                             'multi-label classification')

        if np.isnan(np.sum(X)) or np.isnan(np.sum(y)):
            raise ValueError('Softmax regression does not support '
                             'unknown values')

        self.num_classes = np.unique(y).size
        y = y.ravel().astype(int)
        Y = np.eye(self.num_classes)[y]

        theta = np.zeros(self.num_classes * X.shape[1])
        theta, j, ret = fmin_l_bfgs_b(self.cost_grad, theta,
                                      args=(X, Y, y), **self.fmin_args)
        Theta = theta.reshape((self.num_classes, X.shape[1]))

        return SoftmaxRegressionClassifier(Theta)


class SoftmaxRegressionClassifier(classification.Model):
    def __init__(self, Theta):
        self.Theta = Theta

    def predict(self, X):
        P = np.exp(X.dot(self.Theta.T))
        P /= np.sum(P, axis=1)[:, None]
        return P


if __name__ == '__main__':
    import Orange.data
    from sklearn.cross_validation import StratifiedKFold

    def numerical_grad(f, params, e=1e-4):
        grad = np.zeros_like(params)
        perturb = np.zeros_like(params)
        for i in range(params.size):
            perturb[i] = e
            j1 = f(params - perturb)
            j2 = f(params + perturb)
            grad[i] = (j2 - j1) / (2.0 * e)
            perturb[i] = 0
        return grad

    d = Orange.data.Table('iris')
    m = SoftmaxRegressionLearner(lambda_=1.0)

#    # gradient check
#    m.num_classes = 3
#    Theta = np.random.randn(3 * 4)
#    y = d.Y.ravel().astype(int)
#    Y = np.eye(3)[y]
#
#    ga = m.cost_grad(Theta, d.X, Y, y)[1]
#    gn = numerical_grad(lambda t: m.cost_grad(t, d.X, Y, y)[0], Theta)
#    print(np.sum((ga - gn)**2))

    for lambda_ in [0.1, 0.3, 1, 3, 10]:
        m = SoftmaxRegressionLearner(lambda_=lambda_)
        scores = []
        for tr_ind, te_ind in StratifiedKFold(d.Y.ravel()):
            s = np.mean(m(d[tr_ind])(d[te_ind]) == d[te_ind].Y.ravel())
            scores.append(s)
        print('{:4.1f} {}'.format(lambda_, np.mean(scores)))
