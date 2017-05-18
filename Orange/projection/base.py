import inspect
import threading

import Orange.data
from Orange.base import _ReprableWithPreprocessors
from Orange.misc.wrapper_meta import WrapperMeta
import Orange.preprocess

__all__ = ["Projector", "Projection", "SklProjector"]


class Projector(_ReprableWithPreprocessors):
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

    def __call__(self, data):
        data = self.preprocess(data)
        self.domain = data.domain
        clf = self.fit(data.X, data.Y)
        clf.pre_domain = data.domain
        clf.name = self.name
        return clf

    def preprocess(self, data):
        for pp in self.preprocessors:
            data = pp(data)
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
        self.__dict__.update(proj.__dict__)
        self.proj = proj

    def transform(self, X):
        return self.proj.transform(X)

    def __call__(self, data):
        return data.transform(self.domain)

    def __repr__(self):
        return self.name


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
            spec = inspect.getargs(sklprojection.__init__.__code__)
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
