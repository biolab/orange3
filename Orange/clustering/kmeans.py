import numpy as np
import sklearn.cluster as skl_cluster
from sklearn.metrics import silhouette_score

from Orange.data import Table, DiscreteVariable, Domain, Instance
from Orange.projection import SklProjector, Projection
from Orange.distance import Euclidean


__all__ = ["KMeans"]


class KMeans(SklProjector):
    __wraps__ = skl_cluster.KMeans

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = skl_cluster.KMeans(**self.params)
        if isinstance(X, Table):
            proj = proj.fit(X.X, Y)
            proj.silhouette = silhouette_score(X.X, proj.labels_)
        else:
            proj = proj.fit(X, Y)
            proj.silhouette = silhouette_score(X, proj.labels_)
        proj.inertia = proj.inertia_ / len(X)
        cluster_dist = Euclidean(proj.cluster_centers_).X
        proj.inter_cluster = np.mean(cluster_dist[np.triu_indices_from(cluster_dist, 1)])
        return KMeansModel(proj, self.preprocessors)


class KMeansModel(Projection):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj)
        self.k = self.proj.get_params()["n_clusters"]
        self.centroids = self.proj.cluster_centers_

    def __call__(self, data):
        if isinstance(data, Table):
            if data.domain is not self.pre_domain:
                data = Table(self.pkmre_domain, data)
            c = DiscreteVariable(name='Cluster id', values=range(self.k))
            domain = Domain([c])
            return Table(
                domain,
                self.proj.predict(data.X).astype(int).reshape((len(data), 1)))
        elif isinstance(data, Instance):
            if data.domain is not self.pre_domain:
                data = Instance(self.pre_domain, data)
            c = DiscreteVariable(name='Cluster id', values=range(self.k))
            domain = Domain([c])
            return Table(
                domain,
                np.atleast_2d(self.proj.predict(data._x)).astype(int))
        else:
            return self.proj.predict(data).reshape((len(data), 1))
