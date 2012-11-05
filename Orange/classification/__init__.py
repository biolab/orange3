import copy
import numpy as np
import bottleneck as bn
from Orange import data as Orange_data
from ..data.value import Value


class Fitter:
    """
    Return a model build from the data given as a :class:`data.Table`

    Models can be implemented using one or two classes.

    In a two-class model, the first class is derived from `Fitter` and overloads
    the `__call__` method. The method gets a :class:`~Orange.data.Table` and
    returns an instance of the corresponding Model class. The `Model` should
    implement method `predict` (see documentation on :class:`Model`).

    In a single-class model, the class should be derived from :class:`Fitter`
    and :class:`Model` and provide two methods:

    - `fit` gets X, Y, and weights (or `None`) as arguments and initializes the
       model's parameters

    - `predict` (see documentation on :class:`~Orange.classification.Model`).
    """

    def fit(self, X, Y, W):
        raise TypeError("Descendants of Fitter must overload method fit")

    def __call__(self, data):
        X, Y, W = data.X, data.Y, data.W if data.has_weights else None
        if np.shape(Y)[0]!=1 and np.shape(Y)[1]!=1:
            raise ValueError("more than one class variable")
        Y = np.reshape(Y, -1)
        clf = self.fit(X,Y,W)
        clf.domain = data.domain
        clf.used_vals = np.unique(Y)
        return clf


class Model:
    Value = 0
    Probs = 1
    ValueProbs =2


    def __init__(self, domain=None):
        if isinstance(self, Fitter):
            domain = None
        elif not domain:
            raise ValueError("unspecified domain")
        self.domain = domain


    def predict(self, table):
        raise TypeError("Descendants of Model must overload method predict")


    def __call__(self, data, ret=Value):
        # Check if this is an unfitted single-class model
        if isinstance(self, Fitter) and self.domain is None:
            model = copy.deepcopy(self)
            model.fit(data.X, data.Y, data.W if data.has_weights else None)
            model.domain = data.domain
            return model

        if not 0 <= ret <= 2:
            raise ValueError("invalid value of argument 'ret'")
        if ret > 0 and any(isinstance(v, Orange_data.ContinuousVariable)
                           for v in self.domain.class_vars):
            raise ValueError("cannot predict continuous distributions")

        # Call the predictor
        if isinstance(data, np.ndarray):
            prediction = self.predict(np.atleast_2d(data))
        elif isinstance(data, Orange_data.Instance):
            if data.domain != self.domain:
                data = Orange_data.Instance(self.domain, data)
            prediction = self.predict(np.atleast_2d(data._values))
        elif isinstance(data, Orange_data.Table):
            if data.domain != self.domain:
                data = Orange_data.Table.new_from_table(self.domain, data)
            prediction = self.predict(data.X)
        else:
            raise TypeError("Unrecognized argument (instance of '%s')",
                            type(data).__name__)

        # Parse the result into value and probs
        multitarget = len(self.domain.class_vars) > 1
        if isinstance(prediction, tuple):
            value, probs = prediction
        elif prediction.ndim == 1 + multitarget:
            value, probs = prediction, None
        elif prediction.ndim == 2 + multitarget:
            value, probs = None, prediction
        else:
            raise TypeError("model returned a %i-dimensional array",
                            prediction.ndim)

        # Ensure that we have what we need to return
        if ret != Model.Probs and value is None:
            value = np.argmax(probs, axis=1)
        if ret != Model.Value and probs is None:
            if multitarget:
                max_card = max(len(c.values) for c in self.domain.class_vars)
                probs = np.zeros(value.shape + (max_card,), float)
                for i, cvar in enumerate(self.domain.class_vars):
                    probs[i] = bn.bincount(np.atleast_2d(value[:, i]), max_card)
            else:
                probs = bn.bincount(np.atleast_2d(value),
                                    len(self.domain.class_var.values))
            return probs

        # Expand probability predictions for class values which are not present
        if isinstance(self.domain.class_var, Orange_data.DiscreteVariable):
            if len(self.domain.class_var.values) != len(probs):
                probs_ext = np.ndarray((len(value), len(self.domain.class_var.values)))
                i = 0
                for ci, cv in enumerate(self.domain.class_var.values):
                    if i < len(self.used_vals) and cv == self.used_vals[i]:
                        probs_ext[:,ci] = probs[:,i]
                        i += 1
                    else:
                        probs_ext[:,ci] = np.zeros(len(value))
                probs = probs_ext

        # Return what we need to
        if ret == Model.Value:
            if isinstance(data, Orange_data.Instance) and not multitarget:
                value = Value(self.domain.class_var, value[0])
            return value
        if ret == Model.Probs:
            return probs
        else: # ret == Model.ValueProbs
            return value, probs
