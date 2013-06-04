import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b

from Orange import classification


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLPLearner(classification.Fitter):
    def __init__(self, layers, lambda_=1.0, dropout=None, callback=None, 
                 **fmin_args):
        self.layers = layers
        self.lambda_ = lambda_
        self.dropout = dropout
        self.callback = callback
        self.fmin_args = fmin_args

    def unfold_params(self, params):
        T, b = [], []
        acc = 0
        for l1, l2 in zip(self.layers, self.layers[1:]):
            b.append(params[acc:acc + l2])
            acc += l2
            T.append(params[acc:acc + l1 * l2].reshape((l2, l1)))
            acc += l1 * l2
        return T, b

    def cost_grad(self, params, X, Y):
        T, b = self.unfold_params(params)

        # forward pass
        a, z = [], []

        a.append(X)
        for i in range(len(self.layers) - 2):
            z.append(a[i].dot(T[i].T) + b[i])
            a.append(sigmoid(z[i]))

        # softmax last layer
        z.append(a[-1].dot(T[-1].T) + b[-1])
        P = np.exp(z[-1] - np.max(z[-1], axis=1)[:, None])
        P /= np.sum(P, axis=1)[:, None]
        a.append(P)

        # cost
        cost = -np.sum(np.log(a[-1] + 1e-15) * Y)
        for theta in T:
            cost += self.lambda_ * np.dot(theta.flat, theta.flat) / 2.0
        cost /= X.shape[0]

        # gradient
        params = [] 
        for i in range(len(self.layers) - 1):
            if i == 0:
                d = a[-1] - Y
            else:
                d = d.dot(T[-i]) * a[-i - 1] * (1 - a[-i - 1])
            dT = a[-i - 2].T.dot(d).T + self.lambda_ * T[-i - 1]
            db = np.sum(d, axis=0)

            params.extend([dT.flat, db.flat])
        grad = np.concatenate(params[::-1]) / X.shape[0]

        return cost, grad


    def fit(self, X, Y, W):
        if np.isnan(np.sum(X)) or np.isnan(np.sum(Y)):
            raise ValueError('Softmax regression does not support '
                             'unknown values')

        if Y.shape[1] == 1:
            num_classes = np.unique(Y).size
            Y = np.eye(num_classes)[Y.ravel().astype(int)]

        params = []
        num_params = 0
        for l1, l2 in zip(self.layers, self.layers[1:]):
            num_params += l1 * l2 + l2
            i = 4.0 * np.sqrt(6.0 / (l1 + l2))
            params.append(np.random.uniform(-i, i, l1 * l2))
            params.append(np.zeros(l2))

        params = np.concatenate(params)
        params, j, ret = fmin_l_bfgs_b(self.cost_grad, params,
                                      args=(X, Y), **self.fmin_args)
        T, b = self.unfold_params(params)
        return MLPClassifier(T, b)


class MLPClassifier(classification.Model):
    def __init__(self, T, b):
        self.T = T
        self.b = b

    def predict(self, X):
        a = X
        for i in range(len(self.T) - 1):
            a = sigmoid(a.dot(self.T[i].T) + self.b[i])
        z = a.dot(self.T[-1].T) + self.b[-1]
        P = np.exp(z - np.max(z, axis=1)[:, None])
        P /= np.sum(P, axis=1)[:, None]
        return P

if __name__ == '__main__':
    import Orange.data
    from sklearn.cross_validation import StratifiedKFold

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

    d = Orange.data.Table('iris')

#    # gradient check
#    m = MLPLearner([4, 20, 20, 3])
#    params = np.random.randn(5 * 20 + 21 * 20 + 21 * 3) * 0.1
#    Y = np.eye(3)[d.Y.ravel().astype(int)]
#
#    ga = m.cost_grad(params, d.X, Y)[1]
#    gm = numerical_grad(lambda t: m.cost_grad(t, d.X, Y)[0], params)
#
#    print(np.sum((ga - gm)**2))

    for lambda_ in [0.03, 0.1, 0.3, 1, 3, 10]:
        m = MLPLearner([4, 20, 20, 3], lambda_=lambda_)
        scores = []
        for tr_ind, te_ind in StratifiedKFold(d.Y.ravel()):
            s = np.mean(m(d[tr_ind])(d[te_ind]) == d[te_ind].Y.ravel())
            scores.append(s)
        print(np.mean(scores), lambda_)


