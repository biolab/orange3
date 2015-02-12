import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from Orange.classification import Learner, Model

__all__ = ["LinearRegressionLearner"]


class LinearRegressionLearner(Learner):
    def __init__(self, lambda_=1.0, preprocessors=None, **fmin_args):
        '''L2 regularized linear regression (a.k.a Ridge regression)

        This model uses the L-BFGS algorithm to minimize the linear least
        squares penalty with L2 regularization. When using this model you
        should:

        - Choose a suitable regularization parameter lambda_
        - Continuize all discrete attributes
        - Consider appending a column of ones to the dataset (intercept term)
        - Transform the dataset so that the columns are on a similar scale

        :param lambda_: the regularization parameter. Higher values of lambda_
        force the coefficients to be small.
        :type lambda_: float

        Examples
        --------

            import numpy as np
            from Orange.data import Table
            from Orange.classification.linear_regression import LinearRegressionLearner

            data = Table('housing')
            data.X = (data.X - np.mean(data.X, axis=0)) / np.std(data.X, axis=0) # normalize
            data.X = np.hstack((data.X, np.ones((data.X.shape[0], 1)))) # append ones
            m = LinearRegressionLearner(lambda_=1.0)
            c = m(data) # fit
            print(c(data)) # predict
        '''
        super().__init__(preprocessors=preprocessors)
        self.lambda_ = lambda_
        self.fmin_args = fmin_args

    def cost_grad(self, theta, X, y):
        t = X.dot(theta) - y

        cost = t.dot(t)
        cost += self.lambda_ * theta.dot(theta)
        cost /= 2.0 * X.shape[0]

        grad = X.T.dot(t)
        grad += self.lambda_ * theta
        grad /= X.shape[0]

        return cost, grad

    def fit(self, X, Y, W):
        if len(Y.shape) > 1 and Y.shape[1] > 1:
            raise ValueError('Linear regression does not support '
                             'multi-target classification')

        if np.isnan(np.sum(X)) or np.isnan(np.sum(Y)):
            raise ValueError('Linear regression does not support '
                             'unknown values')

        theta = np.zeros(X.shape[1])
        theta, cost, ret = fmin_l_bfgs_b(self.cost_grad, theta,
                                         args=(X, Y.ravel()), **self.fmin_args)

        return LinearRegressionModel(theta)


class LinearRegressionModel(Model):
    def __init__(self, theta):
        self.theta = theta

    def predict(self, X):
        return X.dot(self.theta)


if __name__ == '__main__':
    import Orange.data
    import sklearn.cross_validation as skl_cross_validation
    
    np.random.seed(42)

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

    d = Orange.data.Table('housing')
    d.X = np.hstack((d.X, np.ones((d.X.shape[0], 1))))
    d.shuffle()

#    m = LinearRegressionLearner(lambda_=1.0)
#    print(m(d)(d))

#    # gradient check
#    m = LinearRegressionLearner(lambda_=1.0)
#    theta = np.random.randn(d.X.shape[1])
#
#    ga = m.cost_grad(theta, d.X, d.Y.ravel())[1]
#    gm = numerical_grad(lambda t: m.cost_grad(t, d.X, d.Y.ravel())[0], theta)
#
#    print(np.sum((ga - gm)**2))

    for lambda_ in (0.01, 0.03, 0.1, 0.3, 1, 3):
        m = LinearRegressionLearner(lambda_=lambda_)
        scores = []
        for tr_ind, te_ind in skl_cross_validation.KFold(d.X.shape[0]):
            s = np.mean((m(d[tr_ind])(d[te_ind]) - d[te_ind].Y.ravel())**2)
            scores.append(s)
        print('{:5.2f} {}'.format(lambda_, np.mean(scores)))

    m = LinearRegressionLearner(lambda_=0)
    print('test data', np.mean((m(d)(d) - d.Y.ravel())**2))
    print('majority', np.mean((np.mean(d.Y.ravel()) - d.Y.ravel())**2))
