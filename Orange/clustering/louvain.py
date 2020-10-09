"""Python port for Louvain clustering, available at
https://github.com/taynaud/python-louvain

Original C++ implementation available at
https://sites.google.com/site/findcommunities/
"""

import numpy as np
import networkx as nx
# NOTE: The ``community`` package might be renamed in the near future, see
# GH issue https://github.com/taynaud/python-louvain/issues/23
from community import best_partition
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from Orange.clustering.clustering import Clustering
from Orange.data import Table


__all__ = ["Louvain", "matrix_to_knn_graph"]


def jaccard(x, y):
    # type: (set, set) -> float
    """Compute the Jaccard similarity between two sets."""
    return len(x & y) / len(x | y)


def matrix_to_knn_graph(data, k_neighbors, metric, progress_callback=None):
    """Convert data matrix to a graph using a nearest neighbors approach with
    the Jaccard similarity as the edge weights.

    Parameters
    ----------
    data : np.ndarray
    k_neighbors : int
    metric : str
        A distance metric supported by sklearn.
    progress_callback : Callable[[float], None]

    Returns
    -------
    nx.Graph

    """
    # We do k + 1 because each point is closest to itself, which is not useful
    if metric == "cosine":
        # Cosine distance on row-normalized data has the same ranking as
        # Euclidean distance, so we use the latter, which is more efficient
        # because it uses ball trees. We do not need actual distances. If we
        # would, the N * k distances can be recomputed later.
        data = data / np.linalg.norm(data, axis=1)[:, None]
        metric = "euclidean"
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric).fit(data)
    nearest_neighbors = knn.kneighbors(data, return_distance=False)
    # Convert to list of sets so jaccard can be computed efficiently
    nearest_neighbors = list(map(set, nearest_neighbors))
    num_nodes = len(nearest_neighbors)

    # Create an empty graph and add all the data ids as nodes for easy mapping
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data)))

    for idx, node in enumerate(graph.nodes):
        if progress_callback:
            progress_callback(idx / num_nodes)

        for neighbor in nearest_neighbors[node]:
            graph.add_edge(
                node,
                neighbor,
                weight=jaccard(
                    nearest_neighbors[node], nearest_neighbors[neighbor]),
            )

    return graph


class LouvainMethod(BaseEstimator):

    def __init__(self, k_neighbors=30, metric="l2", resolution=1.0,
                 random_state=None):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.resolution = resolution
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # If we are given a table, we have to convert it to a graph first
        graph = matrix_to_knn_graph(
            X, metric=self.metric, k_neighbors=self.k_neighbors)
        return self.fit_graph(graph)

    def fit_graph(self, graph):
        partition = best_partition(
            graph, resolution=self.resolution, random_state=self.random_state)
        self.labels_ = np.fromiter(
            list(zip(*sorted(partition.items())))[1], dtype=int)
        return self


class Louvain(Clustering):
    """Louvain clustering for community detection in graphs.

    Louvain clustering is a community detection algorithm for detecting
    clusters of "communities" in graphs. As such, tabular data must first
    be converted into graph form. This is typically done by computing the
    KNN graph on the input data.

    Attributes
    ----------
    k_neighbors : Optional[int]
        The number of nearest neighbors to use for the KNN graph if
        tabular data is passed.

    metric : Optional[str]
        The metric to use to compute the nearest neighbors.

    resolution : Optional[float]
        The resolution is a parameter of the Louvain method that affects
        the size of the recovered clusters.

    random_state: Union[int, RandomState]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.
    """

    __wraps__ = LouvainMethod

    def __init__(self, k_neighbors=30, metric="l2", resolution=1.0,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors, vars())

    def get_model(self, data):
        if isinstance(data, nx.Graph):
            return self.__returns__(
                self.__wraps__(**self.params).fit_graph(data))
        else:
            return super().get_model(data)


if __name__ == "__main__":
    # clustering run on iris data - orange table
    d = Table("iris")
    louvain = Louvain(5)
    clusters = louvain(d)
