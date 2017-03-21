import inspect
from collections import Iterable
import re

import numpy as np
import scipy

from Orange.data import Table, Storage, Instance, Value
from Orange.data.util import one_hot
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import (RemoveNaNClasses, Continuize,
                               RemoveNaNColumns, SklImpute, Normalize)
from Orange.util import Reprable, patch

__all__ = ["Learner", "Model", "SklLearner", "SklModel"]


class _ReprableWithPreprocessors(Reprable):
    def __repr__(self):
        preprocessors = self.preprocessors
        # TODO: Implement __eq__ on preprocessors to avoid comparing reprs
        if repr(preprocessors) == repr(list(self.__class__.preprocessors)):
            # If they are default, set to None temporarily to avoid printing
            preprocessors = None
        with patch.object(self, 'preprocessors', preprocessors):
            return super().__repr__()


class _ReprableWithParams(Reprable):
    def __repr__(self):
        # In addition to saving values onto self, SklLearners save into params
        with patch.object(self, '__dict__', dict(self.__dict__, **self.params)):
            return super().__repr__()


class Learner(_ReprableWithPreprocessors):
    """The base learner class.

    Preprocessors can behave in a number of different ways, all of which are
    described here.
    If the user does not pass a preprocessor argument into the Learner
    constructor, the default learner preprocessors are used. We assume the user
    would simply like to get things done without having to worry about
    preprocessors.
    If the user chooses to pass in their own preprocessors, we assume they know
    what they are doing. In this case, only the user preprocessors are used and
    the default preprocessors are ignored.
    In case the user would like to use the default preprocessors as well as
    their own ones, the `use_default_preprocessors` flag should be set.

    Parameters
    ----------
    preprocessors : Preprocessor or tuple[Preprocessor], optional
        User defined preprocessors. If the user specifies their own
        preprocessors, the default ones will not be used, unless the
        `use_default_preprocessors` flag is set.

    Attributes
    ----------
    preprocessors : tuple[Preprocessor] (default None)
        The used defined preprocessors that will be used on any data.
    use_default_preprocessors : bool (default False)
        This flag indicates whether to use the default preprocessors that are
        defined on the Learner class. Since preprocessors can be applied in a
        number of ways
    active_preprocessors : tuple[Preprocessor]
        The processors that will be used when data is passed to the learner.
        This depends on whether the user has passed in their own preprocessors
        and whether the `use_default_preprocessors` flag is set.

        This property is needed mainly because of the `Fitter` class, which can
        not know in advance, which preprocessors it will need to use. Therefore
        this resolves the active preprocessors using a lazy approach.
    params : dict
        The params that the learner is constructed with.

    """
    supports_multiclass = False
    supports_weights = False
    #: A sequence of data preprocessors to apply on data prior to
    #: fitting the model
    preprocessors = ()
    learner_adequacy_err_msg = ''

    def __init__(self, preprocessors=None):
        self.use_default_preprocessors = False
        self.__params = {}
        if isinstance(preprocessors, Iterable):
            self.preprocessors = tuple(preprocessors)
        elif preprocessors:
            self.preprocessors = (preprocessors,)

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, value):
        self.__params = value

    def fit(self, X, Y, W=None):
        raise RuntimeError(
            "Descendants of Learner must overload method fit or fit_storage")

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
        origdata = data
        data = self.preprocess(data)

        if len(data.domain.class_vars) > 1 and not self.supports_multiclass:
            raise TypeError("%s doesn't support multiple class variables" %
                            self.__class__.__name__)

        self.domain = data.domain
        model = self._fit_model(data)
        model.domain = data.domain
        model.supports_multiclass = self.supports_multiclass
        model.name = self.name
        model.original_domain = origdomain
        model.original_data = origdata
        model.used_vals = [np.unique(y) for y in data.Y[:, None].T]
        return model

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            model = self.fit_storage(data)
        else:
            X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
            model = self.fit(X, Y, W)
        model.params = self.params
        return model

    def preprocess(self, data):
        """Apply the `preprocessors` to the data"""
        for pp in self.active_preprocessors:
            data = pp(data)
        return data

    @property
    def active_preprocessors(self):
        yield from self.preprocessors
        if (self.use_default_preprocessors and
                self.preprocessors is not type(self).preprocessors):
            yield from type(self).preprocessors

    def check_learner_adequacy(self, domain):
        return True

    @property
    def name(self):
        """Return a short name derived from Learner type name"""
        try:
            return self.__name
        except AttributeError:
            name = self.__class__.__name__
            if name.endswith('Learner'):
                name = name[:-len('Learner')]
            if name.endswith('Fitter'):
                name = name[:-len('Fitter')]
            if isinstance(self, SklLearner) and name.startswith('Skl'):
                name = name[len('Skl'):]
            name = name or 'learner'
            # From http://stackoverflow.com/a/1176023/1090455 <3
            self.name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2',
                               re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', name)).lower()
            return self.name

    @name.setter
    def name(self, value):
        self.__name = value

    def __str__(self):
        return self.name


class Model(Reprable):
    supports_multiclass = False
    supports_weights = False
    Value = 0
    Probs = 1
    ValueProbs = 2

    def __init__(self, domain=None):
        if isinstance(self, Learner):
            domain = None
        elif domain is None:
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
        # Params represented as a comment because not passed into constructor
        return super().__repr__() + '  # params=' + repr(self.params)


class SklLearner(_ReprableWithParams, Learner, metaclass=WrapperMeta):
    """
    ${skldoc}
    Additional Orange parameters

    preprocessors : list, optional (default=[RemoveNaNClasses(), Continuize(), SklImpute(), RemoveNaNColumns()])
        An ordered list of preprocessors applied to data before
        training or testing.
    """
    __wraps__ = None
    __returns__ = SklModel

    preprocessors = default_preprocessors = [
        RemoveNaNClasses(),
        Continuize(),
        RemoveNaNColumns(),
        SklImpute(),
    ]

    def __init__(self, *args, **kwargs):
        self.__params = {}
        super().__init__(*args, **kwargs)

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, value):
        self.__params = self._get_sklparams(value)

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

        if any(v.is_discrete and len(v.values) > 2 for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support " +
                             "multinomial variables.")

        return data

    def fit(self, X, Y, W=None):
        clf = self.__wraps__(**self.params)
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            return self.__returns__(clf.fit(X, Y))
        return self.__returns__(clf.fit(X, Y, sample_weight=W.reshape(-1)))

    @property
    def supports_weights(self):
        """Indicates whether this learner supports weighted instances.
        """
        return 'sample_weight' in self.__wraps__.fit.__code__.co_varnames


class TreeModel(Model):
    pass


class RandomForestModel(Model):
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


class NNBase:
    """Base class for neural network (classification and regression) learners
    """
    preprocessors = SklLearner.preprocessors + [Normalize()]

    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None,
                 tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-08, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
