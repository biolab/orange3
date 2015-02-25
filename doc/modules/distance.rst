#######################
Distance (``distance``)
#######################

The following example demonstrates how to compute distances between all examples:

    >>> from Orange.data import Table
    >>> from Orange.distance import Euclidean
    >>> iris = Table('iris')
    >>> dist_matrix = Euclidean(iris)
    >>> # Distance between first two examples
    >>> dist_matrix.X[0, 1]
    0.53851648



The module for `Distance` is based on the popular `scikit-learn`_ and `scipy`_ packages. We wrap the following distance metrics:

- :obj:`Orange.distance.Euclidean`
- :obj:`Orange.distance.Manhattan`
- :obj:`Orange.distance.Cosine`
- :obj:`Orange.distance.Jaccard`
- :obj:`Orange.distance.SpearmanR`
- :obj:`Orange.distance.SpearmanRAbsolute`
- :obj:`Orange.distance.PearsonR`
- :obj:`Orange.distance.PearsonRAbsolute`

All distances have a common interface to the __call__ method which is the following:

.. automethod:: Orange.distance.Distance.__call__

.. _`scikit-learn`: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
.. _`scipy`: http://docs.scipy.org/doc/scipy/reference/stats.html
