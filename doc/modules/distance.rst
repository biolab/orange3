#######################
Distance (``distance``)
#######################

The module for `Distance` is based on the popular `scikit-learn`_ and `scipy`_ packages. We wrap the following distance metrics:

- :obj:`Orange.distance.Euclidean`
- :obj:`Orange.distance.Manhattan`
- :obj:`Orange.distance.Cosine`
- :obj:`Orange.distance.Jaccard`
- :obj:`Orange.distance.SpearmanR`
- :obj:`Orange.distance.SpearmanRAbsolute`
- :obj:`Orange.distance.PearsonR`
- :obj:`Orange.distance.PearsonRAbsolute`

.. autoclass:: Orange.distance.SklDistance
    :members: __init__, __call__

.. autoclass:: Orange.distance.SpearmanDistance
    :members: __call__

.. autoclass:: Orange.distance.PearsonDistance
    :members: __call__


Example
=======

    >>> from Orange.data import Table
    >>> from Orange.distance import Euclidean
    >>> iris = Table('iris')
    >>> dist_matrix = Euclidean(iris)
    >>> # Distance between first two examples
    >>> dist_matrix.X[0, 1]
    0.53851648


.. _`scikit-learn`: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
.. _`scipy`: http://docs.scipy.org/doc/scipy/reference/stats.html
