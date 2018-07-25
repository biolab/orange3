"""Python port for Louvain clustering, available at
https://github.com/taynaud/python-louvain

Original C++ implementation available at
https://sites.google.com/site/findcommunities/

"""

import numpy as np
import networkx as nx
from community import best_partition
from sklearn.neighbors import NearestNeighbors

import Orange
from Orange.data import Table


def jaccard(x, y):
    # type: (set, set) -> float
    """Compute the Jaccard similarity between two sets."""
    return len(x & y) / len(x | y)


def table_to_knn_graph(data, k_neighbors, metric, progress_callback=None):
    """Convert tabular data to a graph using a nearest neighbors approach with
    the Jaccard similarity as the edge weights.

    Parameters
    ----------
    data : Table
    k_neighbors : int
    metric : str
        A distance metric supported by sklearn.
    progress_callback : Callable[[float], None]

    Returns
    -------
    nx.Graph

    """
    # We do k + 1 because each point is closest to itself, which is not useful
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric).fit(data.X)
    nearest_neighbors = knn.kneighbors(data.X, return_distance=False)
    # Convert to list of sets so jaccard can be computed efficiently
    nearest_neighbors = list(map(set, nearest_neighbors))
    num_nodes = len(nearest_neighbors)

    # Create an empty graph and add all the data ids as nodes for easy mapping
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data)))

    for idx, node in enumerate(graph.nodes):
        if progress_callback:
            progress_callback(idx / num_nodes)

        for neighbour in nearest_neighbors[node]:
            graph.add_edge(node, neighbour, weight=jaccard(
                nearest_neighbors[node], nearest_neighbors[neighbour]))

    return graph


class Louvain:
    preprocessors = [Orange.preprocess.Continuize(),
                     Orange.preprocess.SklImpute()]

    def __init__(self, k_neighbors=30, metric='l2', resolution=1., preprocessors=None):
        """Louvain clustering for community detection in graphs.

        Louvain clustering is a community detection algorithm for detecting
        clusters of "communities" in graphs. As such, tabular data must first
        be converted into graph form. This is typically done by computing the
        KNN graph on the input data.

        Parameters
        ----------
        k_neighbors : Optional[int]
            The number of nearest neighbors to use for the KNN graph if
            tabular data is passed.
        metric : Optional[str]
            The metric to use to compute the nearest neighbors.
        resolution : Optional[float]
            The resolution is a parameter of the Louvain method that affects
            the size of the recovered clusters.

        """
        if preprocessors is None:
            preprocessors = type(self).preprocessors
        self.preprocessors = tuple(preprocessors)

        self.k_neighbors = k_neighbors
        self.metric = metric
        self.resolution = resolution

        self.labels = None

    def __call__(self, data):
        data = self.preprocess(data)
        return self.fit_predict(data.X, data.Y)

    def preprocess(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data

    def fit(self, X, y=None):
        # If we are given a table, we have to convert it to a graph first
        if isinstance(X, Table):
            graph = table_to_knn_graph(X.X, metric=self.metric, k_neighbors=self.k_neighbors)
        # Same goes for a matrix
        elif isinstance(X, np.ndarray):
            graph = table_to_knn_graph(X, metric=self.metric, k_neighbors=self.k_neighbors)
        elif isinstance(X, nx.Graph):
            graph = X

        partition = best_partition(graph, resolution=self.resolution)
        partition = np.fromiter(list(zip(*sorted(partition.items())))[1], dtype=int)

        self.labels = partition

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels
