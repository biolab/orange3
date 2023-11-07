import warnings

import copy
import inspect
import threading

import numpy as np

import Orange.data
from Orange.base import ReprableWithPreprocessors
from Orange.data.util import SharedComputeValue, get_unique_names
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import RemoveNaNRows
from Orange.util import dummy_callback, wrap_callback, OrangeDeprecationWarning
import Orange.preprocess

__all__ = ["LinearCombinationSql", "Projector", "Projection", "SklProjector",
           "LinearProjector", "DomainProjection"]


class LinearCombinationSql:
    def __init__(self, attrs, weights, mean=None):
        self.attrs = attrs
        self.weights = weights
        self.mean = mean

    def __call__(self):
        if self.mean is None:
            return ' + '.join('{} * {}'.format(w, a.to_sql())
                              for a, w in zip(self.attrs, self.weights))
        return ' + '.join('{} * ({} - {})'.format(w, a.to_sql(), m, w)
                          for a, m, w in zip(self.attrs, self.mean, self.weights))


class Projector(ReprableWithPreprocessors):
    #: A sequence of data preprocessors to apply on data prior to projecting
    name = 'projection'
    preprocessors = ()

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = tuple(preprocessors)
        self.__tls = threading.local()

    def fit(self, X, Y=None):
        raise NotImplementedError(
            "Classes derived from Projector must overload method fit")

    def __call__(self, data, progress_callback=None):
        if progress_callback is None:
            progress_callback = dummy_callback
        progress_callback(0, "Preprocessing...")
        try:
            cb = wrap_callback(progress_callback, end=0.1)
            data = self.preprocess(data, progress_callback=cb)
        except TypeError:
            data = self.preprocess(data)
            warnings.warn("A keyword argument 'progress_callback' has been "
                          "added to the preprocess() signature. Implementing "
                          "the method without the argument is deprecated and "
                          "will result in an error in the future.",
                          OrangeDeprecationWarning, stacklevel=2)
        self.domain = data.domain
        progress_callback(0.1, "Fitting...")
        clf = self.fit(data.X, data.Y)
        clf.pre_domain = data.domain
        clf.name = self.name
        progress_callback(1)
        return clf

    def preprocess(self, data, progress_callback=None):
        if progress_callback is None:
            progress_callback = dummy_callback
        n_pps = len(self.preprocessors)
        for i, pp in enumerate(self.preprocessors):
            progress_callback(i / n_pps)
            data = pp(data)
        progress_callback(1)
        return data

    # Projectors implemented using `fit` access the `domain` through the
    # instance attribute. This makes (or it would) make it impossible to
    # be implemented in a thread-safe manner. So the domain is made a
    # property descriptor utilizing thread local storage behind the scenes.
    @property
    def domain(self):
        return self.__tls.domain

    @domain.setter
    def domain(self, value):
        self.__tls.domain = value

    @domain.deleter
    def domain(self):
        del self.__tls.domain

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_Projector__tls"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__tls = threading.local()


class Projection:
    def __init__(self, proj):
        if proj is not None:
            self.__dict__.update(proj.__dict__)
        self.proj = proj

    def transform(self, X):
        return self.proj.transform(X)

    def __call__(self, data):
        return data.transform(self.domain)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        return type(self) is type(other) \
            and self.proj == other.proj

    def __hash__(self):
        return hash(self.proj)


class TransformDomain:
    def __init__(self, projection):
        self.projection = projection
        self._hash = None

    def __call__(self, data):
        if data.domain != self.projection.pre_domain:
            data = data.transform(self.projection.pre_domain)
        return self.projection.transform(data.X)

    def __eq__(self, other):
        if self is other:
            return True
        return type(self) is type(other) \
            and self.projection == other.projection

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hash = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_hash"]
        return state

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.projection)
        return self._hash


class ComputeValueProjector(SharedComputeValue):
    def __init__(self, projection=None, feature=None, transform=None):
        super().__init__(transform)
        if projection is not None:
            warnings.warn("Argument projection is unused and will be removed.",
                          OrangeDeprecationWarning, stacklevel=2)
        self.projection = projection
        self.feature = feature
        self.transformed = None

    def compute(self, data, space):
        return space[:, self.feature]

    def __eq__(self, other):
        if self is other:
            return True
        return super().__eq__(other) \
            and self.projection == other.projection \
            and self.feature == other.feature \
            and self.transformed == other.transformed

    def __hash__(self):
        return hash((super().__hash__(), self.projection, self.feature, self.transformed))


class DomainProjection(Projection):
    var_prefix = "C"

    def __init__(self, proj, domain, n_components):
        transformer = TransformDomain(self)

        def proj_variable(i, name):
            v = Orange.data.ContinuousVariable(
                name, compute_value=ComputeValueProjector(feature=i, transform=transformer)
            )
            v.to_sql = LinearCombinationSql(
                domain.attributes, self.components_[i, :],
                getattr(self, 'mean_', None))
            return v

        super().__init__(proj=proj)
        self.orig_domain = domain
        self.n_components = n_components
        var_names = self._get_var_names(n_components)
        self.domain = Orange.data.Domain(
            [proj_variable(i, var_names[i]) for i in range(n_components)],
            domain.class_vars, domain.metas)

    def _get_var_names(self, n):
        postfixes = ["x", "y"] if n == 2 else [str(i) for i in range(1, n + 1)]
        names = [f"{self.var_prefix}-{postfix}" for postfix in postfixes]
        return get_unique_names(self.orig_domain, names)

    def copy(self):
        proj = copy.deepcopy(self.proj)
        model = type(self)(proj, self.domain.copy(), self.n_components)
        model.pre_domain = self.pre_domain.copy()
        model.name = self.name
        return model

    def __eq__(self, other):
        # see comment in __hash__() about .domain
        if self is other:
            return True
        return super().__eq__(other) \
            and self.n_components == other.n_components \
            and self.orig_domain == other.orig_domain \
            and self.var_prefix == other.var_prefix

    def __hash__(self):
        # hashing self.domain would cause infinite recursion;
        # because it is only constructed from .orig_domain, .n_components
        # and .proj (dealt with in the superclass), we do not need it
        return hash((super().__hash__(), self.n_components, self.orig_domain, self.var_prefix))


class LinearProjector(Projector):
    name = "Linear Projection"
    supports_sparse = False
    preprocessors = [RemoveNaNRows()]
    projection = DomainProjection

    def __init__(self, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.components_ = None

    def fit(self, X, Y=None):
        self.components_ = self.get_components(X, Y)
        return self.projection(self, self.domain, len(self.components_))

    def get_components(self, X, Y):
        raise NotImplementedError

    def transform(self, X):
        return np.dot(X, self.components_.T)


class SklProjector(Projector, metaclass=WrapperMeta):
    __wraps__ = None
    _params = {}
    name = 'skl projection'
    supports_sparse = False

    preprocessors = [Orange.preprocess.Continuize(),
                     Orange.preprocess.SklImpute()]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = self._get_sklparams(value)

    def _get_sklparams(self, values):
        sklprojection = self.__wraps__
        if sklprojection is not None:
            spec = list(
                inspect.signature(sklprojection.__init__).parameters.keys()
            )
            # first argument is 'self'
            assert spec[0] == "self"
            params = {
                name: values[name] for name in spec[1:] if name in values
            }
        else:
            raise TypeError("Wrapper does not define '__wraps__'")
        return params

    def preprocess(self, data, progress_callback=None):
        data = super().preprocess(data, progress_callback)
        if any(v.is_discrete and len(v.values) > 2
               for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support "
                             "multinomial variables.")
        return data

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        return proj.fit(X, Y)

    def __getattr__(self, item):
        try:
            return self.params[item]
        except (AttributeError, KeyError):
            raise AttributeError(item) from None

    def __dir__(self):
        return list(sorted(set(super().__dir__()) | set(self.params.keys())))
