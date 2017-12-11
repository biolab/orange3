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

    >>> from Orange.projection import PCA
    >>> from Orange.data import Table
    >>> iris = Table('iris')
    >>> pca = PCA()
    >>> model = pca(iris)
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



FreeViz
-------

FreeViz uses a paradigm borrowed from particle physics: points in the same class attract each
other, those from different class repel each other, and the resulting forces are exerted on the
anchors of the attributes, that is, on unit vectors of each of the dimensional axis. The points
cannot move (are projected in the projection space), but the attribute anchors can, so the
optimization process is a hill-climbing optimization where at the end the anchors are placed such
that forces are in equilibrium.


Example
=======

    >>> from Orange.projection import FreeViz
    >>> from Orange.data import Table
    >>> iris = Table('iris')
    >>> freeviz = FreeViz()
    >>> model = freeviz(iris)
    >>> model.components_    # FreeViz components
    array([[  3.83487853e-01,   1.38777878e-17],
       [ -6.95058218e-01,   7.18953457e-01],
       [  2.16525357e-01,  -2.65741729e-01],
       [  9.50450079e-02,  -4.53211728e-01]])
    >>> transformed_data = model(iris)    # transformed data
    >>> transformed_data
    [[-0.157, 2.053 | Iris-setosa],
    [0.114, 1.694 | Iris-setosa],
    [-0.123, 1.864 | Iris-setosa],
    [-0.048, 1.740 | Iris-setosa],
    [-0.265, 2.125 | Iris-setosa],
    ...
    ]



.. autoclass:: Orange.projection.freeviz.FreeViz




LDA
---

Linear discriminant analysis is another way of finding a linear transformation of
data that reduces the number of dimensions required to represent it. It is often
used for dimensionality reduction prior to classification, but can also be used as a
classification technique itself ([1]_).


Example
=======

    >>> from Orange.projection import LDA
    >>> from Orange.data import Table
    >>> iris = Table('iris')
    >>> lda = LDA()
    >>> model = LDA(iris)
    >>> model.components_    # LDA components
    array([[ 0.20490976,  0.38714331, -0.54648218, -0.71378517],
       [ 0.00898234,  0.58899857, -0.25428655,  0.76703217],
       [-0.71507172,  0.43568045,  0.45568731, -0.30200008],
       [ 0.06449913, -0.35780501, -0.42514529,  0.828895  ]])
    >>> transformed_data = model(iris)    # transformed data
    >>> transformed_data
    [[1.492, 1.905 | Iris-setosa],
    [1.258, 1.608 | Iris-setosa],
    [1.349, 1.750 | Iris-setosa],
    [1.180, 1.639 | Iris-setosa],
    [1.510, 1.963 | Iris-setosa],
    ...
    ]



.. autoclass:: Orange.projection.lda.LDA



References
----------

.. [1] Witten, I.H., Frank, E., Hall, M.A. and Pal, C.J., 2016.
   Data Mining: Practical machine learning tools and techniques. Morgan Kaufmann.

