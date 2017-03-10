###########################
Projection (``projection``)
###########################

.. automodule:: Orange.projection



PCA
---

Principal component analysis is a statistical procedure that uses
an orthogonal transformation to convert a set of observations of possibly
correlated variables into a set of values of linearly uncorrelated variables
called principal components.


Example
=======

    >>> from Orange.projection.pca import PCA
    >>> iris = Orange.data.Table('iris')
    >>> pca = PCA()
    >>> model = PCA(iris)
    >>> model.components_    # PCA components
    array([[ 0.36158968, -0.08226889,  0.85657211,  0.35884393],
        [ 0.65653988,  0.72971237, -0.1757674 , -0.07470647],
        [-0.58099728,  0.59641809,  0.07252408,  0.54906091],
        [ 0.31725455, -0.32409435, -0.47971899,  0.75112056]])
    >>> transformed_data = model(iris)    # transformed data
    >>> transformed_data
    [[-2.684, 0.327, -0.022, 0.001 | Iris-setosa],
    [-2.715, -0.170, -0.204, 0.100 | Iris-setosa],
    [-2.890, -0.137, 0.025, 0.019 | Iris-setosa],
    [-2.746, -0.311, 0.038, -0.076 | Iris-setosa],
    [-2.729, 0.334, 0.096, -0.063 | Iris-setosa],
    ...
    ]



.. autoclass:: Orange.projection.pca.PCA
.. autoclass:: Orange.projection.pca.SparsePCA
.. autoclass:: Orange.projection.pca.IncrementalPCA

