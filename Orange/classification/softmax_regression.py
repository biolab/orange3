import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from Orange.classification import Learner, Model
from Orange.data.filter import HasClass
from Orange.preprocess import Continuize, RemoveNaNColumns, Impute, Normalize

__all__ = ["SoftmaxRegressionLearner"]


class SoftmaxRegressionLearner(Learner):
    """L2 regularized softmax regression classifier.
    Uses the L-BFGS algorithm to minimize the categorical
    cross entropy cost with L2 regularization. This model is suitable
    when dealing with a multi-class classification problem.

    When using this learner you should:

    - choose a suitable regularization parameter lambda\_,
    - consider using many logistic regression models (one for each
      value of the class variable) instead of softmax regression.

    Parameters
    ----------

    lambda\_ : float, optional (default=1.0)
        Regularization parameter. It controls trade-off between fitting the
        data and keeping parameters small. Higher values of lambda\_ force
        parameters to be smaller.

    preprocessors : list, optional
        Preprocessors are applied to data before training or testing. Default preprocessors:
        Defaults to
        `[RemoveNaNClasses(), RemoveNaNColumns(), Impute(), Continuize(), Normalize()]`

        - remove columns with all values as NaN
        - replace NaN values with suitable values
        - continuize all discrete attributes,
        - transform the dataset so that the columns are on a similar scale,

    fmin_args : dict, optional
        Parameters for L-BFGS algorithm.
    """
    name = 'softmax'
    preprocessors = [HasClass(),
                     RemoveNaNColumns(),
                     Impute(),
                     Continuize(),
                     Normalize()]

    def __init__(self, lambda_=1.0, preprocessors=None, **fmin_args):
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
        if len(y.shape) > 1:
            raise ValueError('Softmax regression does not support '
                             'multi-label classification')

        if np.isnan(np.sum(X)) or np.isnan(np.sum(y)):
            raise ValueError('Softmax regression does not support '
                             'unknown values')

        X = np.hstack((X, np.ones((X.shape[0], 1))))

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
        X = np.hstack((X, np.ones((X.shape[0], 1))))
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

# for lambda_ in [0.1, 0.3, 1, 3, 10]:
# m = SoftmaxRegressionLearner(lambda_=lambda_)
# scores = []
# for tr_ind, te_ind in StratifiedKFold(d.Y.ravel()):
#            s = np.mean(m(d[tr_ind])(d[te_ind]) == d[te_ind].Y.ravel())
#            scores.append(s)
#        print('{:4.1f} {}'.format(lambda_, np.mean(scores)))
