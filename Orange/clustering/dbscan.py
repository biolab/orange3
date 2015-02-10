import Orange.data
import sklearn.cluster as skl_cluster
from Orange.projection import SklProjection, ProjectionModel
from numpy import atleast_2d, ndarray, where


__all__ = ["DBSCAN"]

class DBSCAN(SklProjection):
    __wraps__ = skl_cluster.DBSCAN

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto',
                 leaf_size=30, p=None, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = skl_cluster.DBSCAN(**self.params)
        self.X = X
        if isinstance(X, Orange.data.Table):
            proj = proj.fit(X.X,)
        else:
            proj = proj.fit(X, )
        return DBSCANModel(proj, self.preprocessors)


class DBSCANModel(ProjectionModel):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj, preprocessors=preprocessors)

    def __call__(self, data):
        data = self.preprocess(data)
        if isinstance(data, ndarray):
            return self.proj.fit_predict(data).reshape((len(data), 1))

        if isinstance(data, Orange.data.Table):
            y = self.proj.fit_predict(data.X)
            vals = [-1] + list(self.proj.core_sample_indices_)
            c = Orange.data.DiscreteVariable(name='Core sample index', values=vals)
            domain = Orange.data.Domain([c])
            return Orange.data.Table(domain, y.reshape(len(y), 1))

        elif isinstance(data, Orange.data.Instance):
            # Instances-by-Instance classification is not defined;
            raise Exception("Core sample assignment is not supported for single instances.")