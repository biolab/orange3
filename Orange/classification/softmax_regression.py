import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from Orange.classification import Learner, Model

__all__ = ["SoftmaxRegressionLearner"]


class SoftmaxRegressionLearner(Learner):
    """L2 regularized softmax regression classifier.
    Uses the L-BFGS algorithm to minimize the categorical
    cross entropy cost with L2 regularization. This model is suitable
    when dealing with a multi-class classification problem

    When using this learner you should:

    - choose a suitable regularization parameter lambda\_,
    - continuize all discrete attributes,
    - consider appending a column of ones to the data set (intercept term),
    - transform the data set so that the columns are on a similar scale,
    - consider using many logistic regression models (one for each
      value of the class variable) instead of softmax regression.

    lambda\_ : float, optional (default=1.0)
       the regularization parameter. Higher values of lambda\_
       force the coefficients to be smaller
    """

    name = 'softmax'

    def __init__(self, lambda_=1.0, normalize=True, preprocessors=None,
                 **fmin_args):
        super().__init__(preprocessors=preprocessors)
        self.lambda_ = lambda_
        self.fmin_args = fmin_args

    def cost_grad(self, Theta_flat, X, Y):
        Theta = Theta_flat.reshape((self.num_classes, X.shape[1]))

        M = X.dot(Theta.T)
        P = np.exp(M - np.max(M, axis=1)[:, None])
        P /= np.sum(P, axis=1)[:, None]

        cost = -np.sum(np.log(P) * Y)
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
        Y = np.eye(self.num_classes)[y.ravel().astype(int)]

        theta = np.zeros(self.num_classes * X.shape[1])
        theta, j, ret = fmin_l_bfgs_b(self.cost_grad, theta,
                                      args=(X, Y), **self.fmin_args)
        Theta = theta.reshape((self.num_classes, X.shape[1]))

        return SoftmaxRegressionModel(Theta)


class SoftmaxRegressionModel(Model):
    def __init__(self, Theta):
        self.Theta = Theta

    def predict(self, X):
        M = X.dot(self.Theta.T)
        P = np.exp(M - np.max(M, axis=1)[:, None])
        P /= np.sum(P, axis=1)[:, None]
        return P


if __name__ == '__main__':
    import Orange.data

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

    # gradient check
    m = SoftmaxRegressionLearner(lambda_=1.0)
    m.num_classes = 3
    Theta = np.random.randn(3 * 4)
    Y = np.eye(3)[d.Y.ravel().astype(int)]

    ga = m.cost_grad(Theta, d.X, Y)[1]
    gn = numerical_grad(lambda t: m.cost_grad(t, d.X, Y)[0], Theta)

    print(ga)
    print(gn)

#    for lambda_ in [0.1, 0.3, 1, 3, 10]:
#        m = SoftmaxRegressionLearner(lambda_=lambda_)
#        scores = []
#        for tr_ind, te_ind in StratifiedKFold(d.Y.ravel()):
#            s = np.mean(m(d[tr_ind])(d[te_ind]) == d[te_ind].Y.ravel())
#            scores.append(s)
#        print('{:4.1f} {}'.format(lambda_, np.mean(scores)))
