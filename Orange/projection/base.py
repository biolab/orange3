import inspect

import Orange.data
import Orange.preprocess
from Orange.classification.base import WrapperMeta

__all__ = ["Projection", "ProjectionModel", "SklProjection"]


class Projection:
    #: A sequence of data preprocessors to apply on data prior to
    #: fitting the model
    name = 'projection'
    preprocessors = ()

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = tuple(preprocessors)

    def fit(self, X, Y=None):
        raise NotImplementedError(
            "Descendants of Projection must overload method fit")

    def __call__(self, data):
        data = self.preprocess(data)
        self.domain = data.domain
        X, Y = data.X, data.Y
        clf = self.fit(X, Y)
        clf.domain = data.domain
        clf.name = self.name
        return clf

    def preprocess(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data


class ProjectionModel:
    def __init__(self, proj, preprocessors=None):
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = tuple(preprocessors)
        self.__dict__.update(proj.__dict__)
        self.proj = proj

    def transform(self, X):
        trns = self.proj.transform(X)
        return trns

    def preprocess(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data

    def __repr__(self):
        return self.name


class SklProjection(Projection, metaclass=WrapperMeta):

    __wraps__ = None
    _params = {}
    name = 'skl projection'

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

        if any(isinstance(v, Orange.data.DiscreteVariable) and len(v.values) > 2
               for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support " +
                             "multinomial variables.")

        return data

    def __call__(self, data):
        proj = super().__call__(data)
        return proj

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        return proj.fit(X, Y)

    def __repr__(self):
        return '{} {}'.format(self.name, self.params)
