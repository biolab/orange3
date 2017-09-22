#######################
Distance (``distance``)
#######################

The following example demonstrates how to compute distances between all data
instances from Iris:

    >>> from Orange.data import Table
    >>> from Orange.distance import Euclidean
    >>> iris = Table('iris')
    >>> dist_matrix = Euclidean(iris)
    >>> # Distance between first two examples
    >>> dist_matrix.X[0, 1]
    0.53851648

To compute distances between all columns, we set `axis` to 0.

    >>> Euclidean(iris, axis=0)
    DistMatrix([[  0.        ,  36.17927584,  28.9542743 ,  57.1913455 ],
                [ 36.17927584,   0.        ,  25.73382987,  25.81259383],
                [ 28.9542743 ,  25.73382987,   0.        ,  33.87270287],
                [ 57.1913455 ,  25.81259383,  33.87270287,   0.        ]])

Finally, we can compute distances between all pairs of rows from two tables.

    >>> iris1 = iris[:100]
    >>> iris2 = iris[100:]
    >>> dist = Euclidean(iris_even, iris_odd)
    >>> dist.shape
    (75, 100)

Most metrics can be fit on training data to normalize values and handle missing
data. We do so by calling the constructor without arguments or with parameters,
such as `normalize`, and then pass the data to method `fit`.

    >>> dist_model = Euclidean(normalize=True).fit(iris1)
    >>> dist = dist_model(iris2[:3])
    >>> dist
    DistMatrix([[ 0.        ,  1.36778277,  1.11352233],
                [ 1.36778277,  0.        ,  1.57810546],
                [ 1.11352233,  1.57810546,  0.        ]])

The above distances are computed on the first three rows of `iris2`, normalized
by means and variances computed from `iris1`.

Here are five closest neighbors of `iris2[0]` from `iris1`::

   >>> dist0 = dist_model(iris1, iris2[0])
   >>> neigh_idx = np.argsort(dist0.flatten())[:5]
   >>> iris1[neigh_idx]
   [[5.900, 3.200, 4.800, 1.800 | Iris-versicolor],
    [6.700, 3.000, 5.000, 1.700 | Iris-versicolor],
    [6.300, 3.300, 4.700, 1.600 | Iris-versicolor],
    [6.000, 3.400, 4.500, 1.600 | Iris-versicolor],
    [6.400, 3.200, 4.500, 1.500 | Iris-versicolor]
   ]

All distances share a common interface.

.. autoclass:: Orange.distance.Distance

Handling discrete and missing data
==================================

Discrete data is handled as appropriate for the particular distance. For
instance, the Euclidean distance treats a pair of values as either the same or
different, contributing either 0 or 1 to the squared sum of differences. In
other cases -- particularly in Jaccard and cosine distance, discrete values
are treated as zero or non-zero.

Missing data is not simply imputed. We assume that values of each variable are
distributed by some unknown distribution and compute - without assuming a
particular distribution shape - the expected distance.
For instance, for the Euclidean distance it turns out that the expected
squared distance between a known and a missing value equals the square of
the known value's distance from the mean of the missing variable, plus its
variance.


Supported distances
===================

Euclidean distance
------------------

For numeric values, the Euclidean distance is the square root of sums of
squares of pairs of values from rows or columns. For discrete values, 1
is added if the two values are different.

To put all numeric data on the same scale, and in particular when working
with a mixture of numeric and discrete data, it is recommended to enable
normalization by adding `normalize=True` to the constructor. With this,
numeric values are normalized by subtracting their mean and divided by
deviation multiplied by the square root of two. The mean and deviation are
computed on the training data, if the `fit` method is used. When computing
distances between two tables and without explicitly calling `fit`, means
and variances are computed from the first table only. Means and variances
are always computed from columns, disregarding the axis over which we
compute the distances, since columns represent variables and hence come from
a certain distribution.

As described above, the expected squared difference between a known and a
missing value equals the squared difference between the known value and the
mean, plus the variance. The squared difference between two unknown values
equals twice the variance.

For normalized data, the difference between a known and missing numeric value
equals the square of the known value + 0.5. The difference between two
missing values is 1.

For discrete data, the expected difference between a known and a missing value
equals the probablity that the two values are different, which is 1 minus the
probability of the known value. If both values are missing, the probability
of them being different equals 1 minus the sum of squares of all probabilities
(also known as the Gini index).


Manhattan distance
------------------

Manhattan distance is the sum of absolute pairwise distances.

Normalization and treatment of missing values is similar as in the Euclidean
distance, except that medians and median absolute distance from the median
(MAD) are used instead of means and deviations.

For discrete values, distances are again 0 or 1, hence the Manhattan distance
for discrete columns is the same as the Euclidean.

Cosine distance
---------------

Cosine similarity is the dot product divided by the product
of lengths (where the length is the square of dot product of a row/column with
itself). Cosine distance is computed by subtracting the similarity from one.

In calculation of dot products, missing values are replaced by means. In
calculation of lengths, the contribution of a missing value equals the square
of the mean plus the variance. (The difference comes from the fact that in
the former case the missing values are independent.)

Non-zero discrete values are replaced by 1. This introduces the notion of a
"base value", which is the first in the list of possible values. In most cases,
this will only make sense for indicator (i.e. two-valued, boolean attributes).

Cosine distance does not support any column-wise normalization.

Jaccard distance
----------------

Jaccard similarity between two sets is defined as the size of their
intersection divided by the size of the union. Jaccard distance is computed
by subtracting the similarity from one.

In Orange, attribute values are interpreted as membership indicator. In
row-wise distances, columns are interpreted as sets, and non-zero
values in a row (including negative values of numeric features) indicate that
the row belongs to the particular sets. In column-wise distances, rows are sets
and values indicate the sets to which the column belongs.

For missing values, relative frequencies from the training data are used as
probabilities for belonging to a set. That is, for row-wise distances, we
compute the relative frequency of non-zero values in each column, and vice-versa
for column-wise distances. For intersection (union) of sets, we then add the
probability of belonging to both (any of) the two sets instead of adding a
0 or 1.

SpearmanR, AbsoluteSpearmanR, PearsonR, AbsolutePearsonR
--------------------------------------------------------

The four correlation-based distance measure equal (1 - the
correlation coefficient) / 2. For `AbsoluteSpearmanR` and `AbsolutePearsonR`, the
absolute value of the coefficient is used.

These distances do not handle missing or discrete values.

Mahalanobis distance
--------------------

Mahalanobis distance is similar to cosine distance, except that the data is
projected into the PCA space.

Mahalanobis distance does not handle missing or discrete values.
