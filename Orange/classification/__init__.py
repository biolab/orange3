import numpy as np
import scipy
import bottlechest as bn

import Orange.data


class Fitter:
    supports_multiclass = False

    def fit(self, X, Y, W=None):
        raise NotImplementedError(
            "Descendants of Fitter must overload method fit")

    def fit_storage(self, data):
        return self.fit(data.X, data.Y, data.W)

    def __call__(self, data):
        if len(data.domain.class_vars) > 1 and not self.supports_multiclass:
            raise TypeError("fitter doesn't support multiple class variables")
        self.domain = data.domain
        if type(self).fit is Fitter.fit:
            clf = self.fit_storage(data)
        else:
            X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
            clf = self.fit(X, Y, W)
        clf.domain = data.domain
        clf.supports_multiclass = self.supports_multiclass
        return clf


class Model:
    supports_multiclass = False
    supports_weights = False
    Value = 0
    Probs = 1
    ValueProbs = 2

    def __init__(self, domain=None):
        if isinstance(self, Fitter):
            domain = None
        elif not domain:
            raise ValueError("unspecified domain")
        self.domain = domain

    def predict(self, X):
        if self.predict_storage == Model.predict_storage:
            raise TypeError("Descendants of Model must overload method predict")
        else:
            Y = np.zeros((len(X), len(self.domain.class_vars)))
            Y[:] = np.nan
            table = Orange.data.Table(self.domain, X, Y)
            return self.predict_storage(table)

    def predict_storage(self, data):
        if isinstance(data, Orange.data.Storage):
            return self.predict(data.X)
        elif isinstance(data, Orange.data.Instance):
            return self.predict(np.atleast_2d(data.x))
        raise TypeError("Unrecognized argument (instance of '{}')".format(
                        type(data).__name__))

    def __call__(self, data, ret=Value):
        if not 0 <= ret <= 2:
            raise ValueError("invalid value of argument 'ret'")
        if (ret > 0
            and any(isinstance(v, Orange.data.ContinuousVariable)
                    for v in self.domain.class_vars)):
            raise ValueError("cannot predict continuous distributions")

        # Call the predictor
        if isinstance(data, np.ndarray):
            prediction = self.predict(np.atleast_2d(data))
        elif isinstance(data, scipy.sparse.csr.csr_matrix):
            prediction = self.predict(data)
        elif isinstance(data, Orange.data.Instance):
            if data.domain != self.domain:
                data = Orange.data.Instance(self.domain, data)
            prediction = self.predict_storage(data)
        elif isinstance(data, Orange.data.Table):
            if data.domain != self.domain:
                data = data.from_table(self.domain, data)
            prediction = self.predict_storage(data)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')".format(
                            type(data).__name__))

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
            value = np.argmax(probs, axis=-1)
        if ret != Model.Value and probs is None:
            if multitarget:
                max_card = max(len(c.values)
                               for c in self.domain.class_vars)
                probs = np.zeros(value.shape + (max_card,), float)
                for i, cvar in enumerate(self.domain.class_vars):
                    probs[:, i, :], _ = bn.bincount(np.atleast_2d(value[:, i]),
                                                    max_card - 1)
            else:
                probs, _ = bn.bincount(np.atleast_2d(value),
                                       len(self.domain.class_var.values) - 1)
            if ret == Model.ValueProbs:
                return value, probs
            else:
                return probs

        # Return what we need to
        if ret == Model.Probs:
            return probs
        if isinstance(data, Orange.data.Instance) and not multitarget:
            value = Orange.data.Value(self.domain.class_var, value[0])
        if ret == Model.Value:
            return value
        else:  # ret == Model.ValueProbs
            return value, probs


class SklModel(Model):
    used_vals = None

    def __init__(self, clf):
        self.clf = clf


    def predict(self, X):
        value = self.clf.predict(X)
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X)
            return value, probs
        return value

    def __call__(self, data, ret=Model.Value):
        prediction = super().__call__(data, ret=ret)

        if ret == Model.Value:
            return prediction

        if ret == Model.Probs:
            probs = prediction
        else:  # ret == Model.ValueProbs
            value, probs = prediction

        # Expand probability predictions for class values which are not present
        if ret != self.Value:
            n_class = len(self.domain.class_vars)
            max_values = max(len(cv.values) for cv in self.domain.class_vars)
            if max_values != probs.shape[-1]:
                if not self.supports_multiclass:
                    probs = probs[:, np.newaxis, :]
                probs_ext = np.zeros((len(probs), n_class, max_values))
                for c in range(n_class):
                    i = 0
                    class_values = len(self.domain.class_vars[c].values)
                    for cv in range(class_values):
                        if (i < len(self.used_vals[c]) and
                                cv == self.used_vals[c][i]):
                            probs_ext[:, c, cv] = probs[:, c, i]
                            i += 1
                if self.supports_multiclass:
                    probs = probs_ext
                else:
                    probs = probs_ext[:, 0, :]

        if ret == Model.Probs:
            return probs
        else:  # ret == Model.ValueProbs
            return value, probs


class SklFitter(Fitter):

    __wraps__ = None
    __returns__ = SklModel
    _params = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = dict(value)
        self._params.pop("self", None)

    def __call__(self, data):
        if any(isinstance(v, Orange.data.DiscreteVariable) and len(v.values) > 2
               for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support " +
                             "multinomial variables.")
        clf = super().__call__(data)
        clf.used_vals = [np.unique(y) for y in data.Y.T]
        return clf

    def fit(self, X, Y, W):
        clf = self.__wraps__(**self.params)
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            return self.__returns__(clf.fit(X, Y))
        return self.__returns__(clf.fit(X, Y, sample_weight=W.reshape(-1)))



