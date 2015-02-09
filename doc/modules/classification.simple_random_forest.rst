.. py:currentmodule:: Orange.classification.simple_random_forest

#######################
Simple Random Forest (``simple_random_forest``)
#######################

.. index:: simple random forest classifier

Random forest built with :obj:`SimpleTreeLearner` trees.

Example
=======

    >>> import Orange
    >>> data = Orange.data.Table('iris')
    >>> lrn = Orange.classification.SimpleRandomForestLearner()
    >>> clf = lrn(data)
    >>> clf(data[:3], clf.Probs)
    array([[ 0.92733333,  0.03066667,  0.042     ],
           [ 0.92733333,  0.03066667,  0.042     ],
           [ 0.92733333,  0.03066667,  0.042     ]])

.. autoclass:: SimpleRandomForestLearner

    .. automethod:: __init__
