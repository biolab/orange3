.. py:currentmodule:: Orange.clustering.hierarchical

###############################
Hierarchical (``hierarchical``)
###############################

.. index:: hierarchical clustering
   pair: clustering; hierarchical clustering

Example
=======

The following example shows clustering of the Iris data with distance
matrix computed with the :obj:`Orange.distance.Euclidean` distance
and clustering using average linkage.

    >>> from Orange import data, distance
    >>> from Orange.clustering import hierarchical
    >>> data = data.Table('iris')
    >>> dist_matrix = distance.Euclidean(data)
    >>> hierar = hierarchical.HierarchicalClustering(n_clusters=3)
    >>> hierar.linkage = hierarchical.AVERAGE
    >>> hierar.fit(dist_matrix)
    >>> hierar.labels
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  2.,  2.,
            2.,  2.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  2.,  2.,
            2.,  2.,  0.,  2.,  0.,  2.,  0.,  2.,  2.,  0.,  0.,  2.,  2.,
            2.,  2.,  2.,  0.,  2.,  2.,  2.,  2.,  0.,  2.,  2.,  2.,  0.,
            2.,  2.,  2.,  0.,  2.,  2.,  0.])


Hierarchical Clustering
-----------------------

.. autoclass:: HierarchicalClustering
   :members:
