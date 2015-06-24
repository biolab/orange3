import inspect

import numpy as np
import scipy
import bottlechest as bn

from Orange.data import Table, Storage, Instance, Value
from Orange.preprocess import Continuize, RemoveNaNColumns, SklImpute
from Orange.misc.wrapper_meta import WrapperMeta

__all__ = ["Learner", "Model", "SklLearner", "SklModel"]


class Learner:
    supports_multiclass = False
    #: A sequence of data preprocessors to apply on data prior to
    #: fitting the model
    name = 'learner'
    preprocessors = ()

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = list(preprocessors)

    def fit(self, X, Y, W=None):
        raise NotImplementedError(
            "Descendants of Learner must overload method fit")

    def fit_storage(self, data):
        return self.fit(data.X, data.Y, data.W)

    def __call__(self, data):
        if isinstance(data, Instance):
            data = Table(data.domain, [data])
        data = self.preprocess(data)

        if len(data.domain.class_vars) > 1 and not self.supports_multiclass:
            raise TypeError("%s doesn't support multiple class variables" %
                            self.__class__.__name__)

        self.domain = data.domain

        if type(self).fit is Learner.fit:
            model = self.fit_storage(data)
        else:
            X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
            model = self.fit(X, Y, W)
        model.domain = data.domain
        model.supports_multiclass = self.supports_multiclass
        model.name = self.name
        return model

    def preprocess(self, data):
        """
        Apply the `preprocessors` to the data.
        """
        for pp in self.preprocessors:
            data = pp(data)
        return data

    def __repr__(self):
        return self.name


class Model:
    supports_multiclass = False
    supports_weights = False
    Value = 0
    Probs = 1
    ValueProbs = 2

    def __init__(self, domain=None):
        if isinstance(self, Learner):
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
            table = Table(self.domain, X, Y)
            return self.predict_storage(table)

    def predict_storage(self, data):
        if isinstance(data, Storage):
            return self.predict(data.X)
        elif isinstance(data, Instance):
            return self.predict(np.atleast_2d(data.x))
        raise TypeError("Unrecognized argument (instance of '{}')".format(
                        type(data).__name__))

    def __call__(self, data, ret=Value):
        if not 0 <= ret <= 2:
            raise ValueError("invalid value of argument 'ret'")
        if (ret > 0
            and any(v.is_continuous for v in self.domain.class_vars)):
            raise ValueError("cannot predict continuous distributions")

        # Call the predictor
        if isinstance(data, np.ndarray):
            prediction = self.predict(np.atleast_2d(data))
        elif isinstance(data, scipy.sparse.csr.csr_matrix):
            prediction = self.predict(data)
        elif isinstance(data, Instance):
            if data.domain != self.domain:
                data = Instance(self.domain, data)
            data = Table(data.domain, [data])
            prediction = self.predict_storage(data)
        elif isinstance(data, Table):
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
        if isinstance(data, Instance) and not multitarget:
            value = Value(self.domain.class_var, value[0])
        if ret == Model.Value:
            return value
        else:  # ret == Model.ValueProbs
            return value, probs

    def __repr__(self):
        return self.name


class SklModel(Model, metaclass=WrapperMeta):
    used_vals = None

    def __init__(self, skl_model):
        self.skl_model = skl_model

    def predict(self, X):
        value = self.skl_model.predict(X)
        if hasattr(self.skl_model, "predict_proba"):
            probs = self.skl_model.predict_proba(X)
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

    def __repr__(self):
        return '{} {}'.format(self.name, self.params)


class SklLearner(Learner, metaclass=WrapperMeta):
    """
    ${skldoc}
    Additional Orange parameters

    preprocessors : list, optional (default=[Continuize(), SklImpute(), RemoveNaNColumns()])
        An ordered list of preprocessors applied to data before
        training or testing.
    """
    __wraps__ = None
    __returns__ = SklModel
    _params = None

    name = 'skl learner'
    preprocessors = [Continuize(),
                     RemoveNaNColumns(),
                     SklImpute(force=False)]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = self._get_sklparams(value)

    def _get_sklparams(self, values):
        skllearner = self.__wraps__
        if skllearner is not None:
            spec = inspect.getargs(skllearner.__init__.__code__)
            # first argument is 'self'
            assert spec.args[0] == "self"
            params = {name: values[name] for name in spec.args[1:]
                      if name in values}
        else:
            raise TypeError("Wrapper does not define '__wraps__'")
        return params

    def preprocess(self, data):
        data = super().preprocess(data)

        if any(v.is_discrete and len(v.values) > 2
               for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support " +
                             "multinomial variables.")

        return data

    def __call__(self, data):
        m = super().__call__(data)
        m.used_vals = [np.unique(y) for y in data.Y[:, None].T]
        m.params = self.params
        return m

    def fit(self, X, Y, W):
        clf = self.__wraps__(**self.params)
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            return self.__returns__(clf.fit(X, Y))
        return self.__returns__(clf.fit(X, Y, sample_weight=W.reshape(-1)))

    def __repr__(self):
        return '{} {}'.format(self.name, self.params)
