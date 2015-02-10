import Orange.data
import sklearn.cluster as skl_cluster
from Orange.projection import SklProjection, ProjectionModel
from numpy import atleast_2d


__all__ = ["KMeans"]


class KMeans(SklProjection):
    __wraps__ = skl_cluster.KMeans

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                  random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = skl_cluster.KMeans(**self.params)
        if isinstance(X, Orange.data.Table):
            proj = proj.fit(X.X, Y)
        else:
            proj = proj.fit(X, Y)
        return KMeansModel(proj, self.preprocessors)


class KMeansModel(ProjectionModel):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj, preprocessors=preprocessors)

    def __call__(self, data):
        data = self.preprocess(data)
        if isinstance(data, Orange.data.Table):
            c = Orange.data.DiscreteVariable(name='Cluster id', values=range(self.proj.get_params()["n_clusters"]))
            domain = Orange.data.Domain([c])
            return Orange.data.Table(domain, self.proj.predict(data.X).astype(int).reshape((len(data), 1)))
        elif isinstance(data, Orange.data.Instance):
            c = Orange.data.DiscreteVariable(name='Cluster id', values=range(self.proj.get_params()["n_clusters"]))
            domain = Orange.data.Domain([c])
            return Orange.data.Table(domain, atleast_2d(self.proj.predict(data._x)).astype(int))
        else:
            return atleast_2d(self.proj.predict(data)).reshape((len(data), 1))
