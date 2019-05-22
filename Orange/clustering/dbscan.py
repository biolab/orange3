import sklearn.cluster

from Orange.clustering.clustering import Clustering
from Orange.data import Table


__all__ = ["DBSCAN"]


class DBSCAN(Clustering):

    __wraps__ = sklearn.cluster.DBSCAN

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None, preprocessors=None):
        super().__init__(preprocessors, vars())


if __name__ == "__main__":
    d = Table("iris")
    km = DBSCAN(preprocessors=None)
    clusters = km(d)
