import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from Orange.classification import Learner, Model

__all__ = ["MLPLearner"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLPLearner(Learner):
    """Multilayer perceptron (feedforward neural network)

    This model uses stochastic gradient descent and the
    backpropagation algorithm to train the weights of a feedforward
    neural network.  The network uses the sigmoid activation
    functions, except for the last layer which computes the softmax
    activation function. The network can be used for binary and
    multiclass classification. Stochastic gradient descent minimizes
    the L2 regularize categorical crossentropy cost function. The
    topology of the network can be customized by setting the layers
    attribute. When using this model you should:

    - Choose a suitable:
        * topology (layers)
        * regularization parameter (lambda\_)
        * dropout (values of 0.2 for the input layer and 0.5 for the hidden
          layers usually work well)
        * The number of epochs of stochastic gradient descent (num_epochs)
        * The learning rate of stochastic gradient descent (learning_rate)
    - Continuize all discrete attributes
    - Transform the data set so that the columns are on a similar scale

    layers : list
        The topology of the network. A network with
        layer=[10, 100, 100, 3] has two hidden layers with 100 neurons each,
        10 features and a class value with 3 distinct values.

    lambda\_ : float, optional (default = 1.0)
        The regularization parameter. Higher values of lambda\_
        force the coefficients to be small.

    dropout : list, optional (default = None)
        The dropout rate for each, but the last,
        layer. The list should have one element less then the parameter layers.
        Values of 0.2 for the input layer and 0.5 for the hidden layers usually
        work well.

    num_epochs : int, optional
        The number of epochs of stochastic gradient descent

    learning_rate : float, optional
        The learning rate of stochastic gradient descent

    batch_size : int, optional
        The batch size of stochastic gradient descent
    """

    name = 'mlp'

    def __init__(self, layers, lambda_=1.0, dropout=None, preprocessors=None,
                 **opt_args):
        super().__init__(preprocessors=preprocessors)
        if dropout is None:
            dropout = [0] * (len(layers) - 1)
        assert len(dropout) == len(layers) - 1

        self.layers = layers
        self.lambda_ = lambda_
        self.dropout = dropout
        self.opt_args = opt_args

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

        dropout_mask = []
        for i in range(len(T)):
            if self.dropout is None or self.dropout[0] < 1e-7:
                dropout_mask.append(1)
            else:
                dropout_mask.append(
                    np.random.binomial(1, 1 - self.dropout[0],
                                       (X.shape[0], self.layers[i])))

        a.append(X * dropout_mask[0])
        for i in range(len(self.layers) - 2):
            z.append(a[i].dot(T[i].T) + b[i])
            a.append(sigmoid(z[i]) * dropout_mask[i + 1])

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
            dT = (a[-i - 2] * dropout_mask[-i - 1]).T.dot(d).T + self.lambda_\
                * T[-i - 1]
            db = np.sum(d, axis=0)

            params.extend([dT.flat, db.flat])
        grad = np.concatenate(params[::-1]) / X.shape[0]

        return cost, grad

    def fit_bfgs(self, params, X, Y):
        params, j, ret = fmin_l_bfgs_b(self.cost_grad, params,
                                       args=(X, Y), **self.opt_args)
        return params

    def fit_sgd(self, params, X, Y, num_epochs=1000, batch_size=100,
                learning_rate=0.1):
        # shuffle examples
        inds = np.random.permutation(X.shape[0])
        X = X[inds]
        Y = Y[inds]

        # split training and validation set
        num_tr = int(X.shape[0] * 0.8)

        X_tr, Y_tr = X[:num_tr], Y[:num_tr]
        X_va, Y_va = X[num_tr:], Y[num_tr:]

        early_stop = 100

        best_params = None
        best_cost = np.inf

        for epoch in range(num_epochs):
            for i in range(0, num_tr, batch_size):
                cost, grad = self.cost_grad(params, X_tr, Y_tr)
                params -= learning_rate * grad

            # test on validation set
            T, b = self.unfold_params(params)
            P_va = MLPModel(T, b, self.dropout).predict(X_va)
            cost = -np.sum(np.log(P_va + 1e-15) * Y_va)

            if cost < best_cost:
                best_cost = cost
                best_params = np.copy(params)
                early_stop *= 2

            if epoch > early_stop:
                break

        return params

    def fit(self, X, Y, W):
        if np.isnan(np.sum(X)) or np.isnan(np.sum(Y)):
            raise ValueError('MLP does not support unknown values')

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

        #params = self.fit_bfgs(params, X, Y)
        params = self.fit_sgd(params, X, Y, **self.opt_args)

        T, b = self.unfold_params(params)
        return MLPModel(T, b, self.dropout)


class MLPModel(Model):
    def __init__(self, T, b, dropout):
        self.T = T
        self.b = b
        self.dropout = dropout

    def predict(self, X):
        a = X
        for i in range(len(self.T) - 1):
            d = 1 - self.dropout[i]
            a = sigmoid(a.dot(self.T[i].T * d) + self.b[i])
        z = a.dot(self.T[-1].T) + self.b[-1]
        P = np.exp(z - np.max(z, axis=1)[:, None])
        P /= np.sum(P, axis=1)[:, None]
        return P

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

    d = Orange.data.Table('iris')

#    # gradient check
#    m = MLPLearner([4, 20, 20, 3], dropout=[0.0, 0.5, 0.0], lambda_=0.0)
#    params = np.random.randn(5 * 20 + 21 * 20 + 21 * 3) * 0.1
#    Y = np.eye(3)[d.Y.ravel().astype(int)]
#
#    ga = m.cost_grad(params, d.X, Y)[1]
#    gm = numerical_grad(lambda t: m.cost_grad(t, d.X, Y)[0], params)
#
#    print(np.sum((ga - gm)**2))

#    m = MLPLearner([4, 20, 3], dropout=[0.0, 0.0], lambda_=1.0)
#    m(d)

    for lambda_ in [0.03, 0.1, 0.3, 1, 3]:
        m = MLPLearner([4, 20, 20, 3], lambda_=lambda_, num_epochs=1000,
                       learning_rate=0.1)
        scores = []
        for tr_ind, te_ind in skl_cross_validation.StratifiedKFold(d.Y.ravel()):
            s = np.mean(m(d[tr_ind])(d[te_ind]) == d[te_ind].Y.ravel())
            scores.append(s)
        print(np.mean(scores), lambda_)
