import numpy as np

from Orange.preprocess.preprocess import RemoveNaNRows, Continuize, Normalize
from Orange.projection import LinearProjector, DomainProjection

__all__ = ["RadViz"]


class RadVizModel(DomainProjection):
    var_prefix = "radviz"


class RadViz(LinearProjector):
    name = "RadViz"
    supports_sparse = False
    preprocessors = [RemoveNaNRows(),
                     Continuize(multinomial_treatment=Continuize.FirstAsBase),
                     Normalize(norm_type=Normalize.NormalizeBySpan)]
    projection = RadVizModel

    def __call__(self, data):
        if data is not None:
            if len([attr for attr in data.domain.attributes
                    if attr.is_discrete and len(attr.values) > 2]):
                raise ValueError("Can not handle categorical variables"
                                 " with more than two values")
        return super().__call__(data)

    def get_components(self, X, Y):
        return np.array([
            (np.cos(t), np.sin(t)) for t in
            [2.0 * np.pi * (i / X.shape[1]) for i in range(X.shape[1])]]).T

    def transform(self, X):
        table = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            row = np.repeat(np.expand_dims(X[i], axis=1), 2, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                s = X[i].sum()
                table[i] = np.divide((self.components_.T * row).sum(axis=0),
                                     s, where=s != 0)
        return table
