import inspect

import Orange.data
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.misc.cache import single_cache
import Orange.preprocess

__all__ = ["Projector", "Projection", "SklProjector"]


class Projector:
    #: A sequence of data preprocessors to apply on data prior to projecting
    name = 'projection'
    preprocessors = ()

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = tuple(preprocessors)

    def fit(self, X, Y=None):
        raise NotImplementedError(
            "Classes derived from Projector must overload method fit")

    def __call__(self, data):
        data = self.preprocess(data)
        self.domain = data.domain
        clf = self.fit(data.X, data.Y)
        clf.pre_domain = self.domain
        clf.name = self.name
        return clf

    def preprocess(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data


class Projection:
    def __init__(self, proj):
        self.__dict__.update(proj.__dict__)
        self.proj = proj

    @single_cache
    def transform(self, X):
        return self.proj.transform(X)

    def __call__(self, data):
        return data.from_table(self.domain, data)

    def __repr__(self):
        return self.name


class SklProjector(Projector, metaclass=WrapperMeta):
    __wraps__ = None
    _params = {}
    name = 'skl projection'
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

    def __repr__(self):
        return '{} {}'.format(self.name, self.params)
