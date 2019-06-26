import warnings

import sklearn.cluster

from Orange.clustering.clustering import Clustering, ClusteringModel
from Orange.data import Table


__all__ = ["KMeans"]


class KMeansModel(ClusteringModel):

    def __init__(self, projector):
        super().__init__(projector)
        self.centroids = projector.cluster_centers_
        self.k = projector.get_params()["n_clusters"]

    def predict(self, X):
        return self.projector.predict(X)


class KMeans(Clustering):

    __wraps__ = sklearn.cluster.KMeans
    __returns__ = KMeansModel

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, random_state=None, preprocessors=None,
                 compute_silhouette_score=None):
        if compute_silhouette_score is not None:
            warnings.warn(
                "compute_silhouette_score is deprecated. Please use "
                "sklearn.metrics.silhouette_score to compute silhouettes.",
                DeprecationWarning)
        super().__init__(
            preprocessors, {k: v for k, v in vars().items()
                            if k != "compute_silhouette_score"})


if __name__ == "__main__":
    d = Table("iris")
    km = KMeans(preprocessors=None, n_clusters=3)
    clusters = km(d)
    model = km.fit_storage(d)
