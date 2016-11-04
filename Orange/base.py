import inspect

import numpy as np
import scipy

from Orange.data import Table, Storage, Instance, Value
from Orange.data.util import one_hot
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import (RemoveNaNClasses, Continuize,
                               RemoveNaNColumns, SklImpute)

__all__ = ["Learner", "Model", "SklLearner", "SklModel"]


class Learner:
    supports_multiclass = False
    supports_weights = False
    name = 'learner'
    #: A sequence of data preprocessors to apply on data prior to
    #: fitting the model
    preprocessors = ()
    learner_adequacy_err_msg = ''

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = list(preprocessors)

    def fit(self, X, Y, W=None):
        raise RuntimeError(
            "Descendants of Learner must overload method fit or "
            "fit_storage")

    def fit_storage(self, data):
        """Default implementation of fit_storage defaults to calling fit.
        Derived classes must define fit_storage or fit"""
        X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
        return self.fit(X, Y, W)

    def __call__(self, data):
        if not self.check_learner_adequacy(data.domain):
            raise ValueError(self.learner_adequacy_err_msg)

        origdomain = data.domain

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
        model.original_domain = origdomain
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

    def check_learner_adequacy(self, domain):
        return True


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
        if type(self).predict_storage is Model.predict_storage:
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
        raise TypeError("Unrecognized argument (instance of '{}')"
                        .format(type(data).__name__))

    def __call__(self, data, ret=Value):
        if not 0 <= ret <= 2:
            raise ValueError("invalid value of argument 'ret'")
        if ret > 0 and any(v.is_continuous for v in self.domain.class_vars):
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
        elif isinstance(data, (list, tuple)):
            if not isinstance(data[0], (list, tuple)):
                data = [data]
            data = Table(self.original_domain, data)
            data = Table(self.domain, data)
            prediction = self.predict_storage(data)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')"
                            .format(type(data).__name__))

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
                    probs[:, i, :] = one_hot(value[:, i])
            else:
                probs = one_hot(value)
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

    def __repr__(self):
        return '{} {}'.format(self.name, self.params)


class SklLearner(Learner, metaclass=WrapperMeta):
    """
    ${skldoc}
    Additional Orange parameters

    preprocessors : list, optional (default=[RemoveNaNClasses(), Continuize(), SklImpute(), RemoveNaNColumns()])
        An ordered list of preprocessors applied to data before
        training or testing.
    """
    __wraps__ = None
    __returns__ = SklModel
    _params = {}

    name = 'skl learner'
    preprocessors = [RemoveNaNClasses(),
                     Continuize(),
                     RemoveNaNColumns(),
                     SklImpute()]

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

    def fit(self, X, Y, W=None):
        clf = self.__wraps__(**self.params)
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            return self.__returns__(clf.fit(X, Y))
        return self.__returns__(clf.fit(X, Y, sample_weight=W.reshape(-1)))

    def __str__(self):
        return '{} {}'.format(self.name, self.params)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ", ".join("{}={}".format(k, v)
                                         for k, v in self.params.items()))

    @property
    def supports_weights(self):
        """Indicates whether this learner supports weighted instances.
        """
        return 'sample_weight' in self.__wraps__.fit.__code__.co_varnames


class RandomForest:
    """Interface for random forest models
    """

    @property
    def trees(self):
        """Return a list of Trees in the forest

        Returns
        -------
        List[Tree]
        """


class KNNBase:
    """Base class for KNN (classification and regression) learners
    """
    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform",
                 algorithm='auto', metric_params=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y, W=None):
        if self.params["metric_params"] is None and \
                        self.params.get("metric") == "mahalanobis":
            self.params["metric_params"] = {"V": np.cov(X.T)}
        return super().fit(X, Y, W)
