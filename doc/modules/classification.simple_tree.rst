.. py:currentmodule:: Orange.classification.simple_tree

#######################
Simple Tree (``simple_tree``)
#######################

.. index:: simple tree classifier

:obj:`SimpleTreeLearner` is an implementation of regression and classification
trees. It uses gain ratio for classification and mean square error for
regression. :obj:`SimpleTreeLearner` was developed for speeding up random
forest construction, but can also be used as a standalone tree learner.

Example
=======

    >>> import Orange
    >>> data = Orange.data.Table('iris')
    >>> lrn = Orange.classification.SimpleTreeLearner()
    >>> clf = lrn(data)
    >>> clf(data[:3], clf.Probs)
    array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.]])

.. autoclass:: SimpleTreeLearner

    .. automethod:: __init__
