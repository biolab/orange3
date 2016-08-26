import numpy as np
import sklearn.cluster as skl_cluster
from sklearn.metrics import silhouette_score

from Orange.data import Table, TableSeries, DiscreteVariable, Domain
from Orange.projection import SklProjector, Projection
from Orange.distance import Euclidean


__all__ = ["KMeans"]


class KMeans(SklProjector):
    __wraps__ = skl_cluster.KMeans

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, random_state=None, preprocessors=None,
                 compute_silhouette_score=False):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self._compute_silhouette = compute_silhouette_score

    def fit(self, X, Y=None):
        proj = skl_cluster.KMeans(**self.params)
        proj = proj.fit(X, Y)
        proj.silhouette = np.nan
        try:
            if self._compute_silhouette and 2 <= proj.n_clusters < X.shape[0]:
                proj.silhouette = silhouette_score(X, proj.labels_)
        except MemoryError:  # Pairwise dist in silhouette fails for large data
            pass
        proj.inertia = proj.inertia_ / X.shape[0]
        cluster_dist = Euclidean(proj.cluster_centers_)
        proj.inter_cluster = np.mean(cluster_dist[np.triu_indices_from(cluster_dist, 1)])
        return KMeansModel(proj, self.preprocessors)


class KMeansModel(Projection):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj)
        self.k = self.proj.get_params()["n_clusters"]
        self.centroids = self.proj.cluster_centers_

    def __call__(self, data):
        # convert to a table so everything works the same
        if isinstance(data, TableSeries):
            new_table = Table(data.domain)
            new_table = Table.concatenate([new_table, data], axis=0, reindex=False)
            data = new_table

        if isinstance(data, Table):
            if data.domain is not self.pre_domain:
                data = Table(self.pre_domain, data)
            c = DiscreteVariable(name='Cluster id', values=range(self.k))
            domain = Domain([c])
            return Table(domain, self.proj.predict(data.X).astype(int).reshape((len(data), 1)))
        else:
            return self.proj.predict(data).reshape((len(data), 1))
