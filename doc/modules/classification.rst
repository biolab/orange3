###################################
Classification (``classification``)
###################################

.. automodule:: Orange.classification

Logistic Regression
-------------------

Naive Bayes
-----------
.. index:: Naive Bayes classifier
.. autoclass:: NaiveBayesLearner
   :members:

:obj:`Orange.classification.NaiveBayesLearner` is based on
the `scikit-learn`_ package. Continuous attributes are discretized
with default discretizer (see TODO), for alternative discretization
technique use discretization preprocessor.

The following code loads lenses data set (four discrete attributes and discrete
class), constructs naive Bayesian learner, uses it on the entire data set
to construct a classifier, and then applies classifier to the first three
data instances:

    >>> import Orange
    >>> lenses = Orange.data.Table('lenses')
    >>> nb = Orange.classification.NaiveBayesLearner()
    >>> classifier = nb(lenses)
    >>> classifier(lenses[0:3], True)
    array([[ 0.04358755,  0.82671726,  0.12969519],
           [ 0.17428279,  0.20342097,  0.62229625],
           [ 0.18633359,  0.79518516,  0.01848125]])

For data sets that include continuous attributes,

.. _`Naive Bayes`: http://en.wikipedia.org/wiki/Naive_Bayes_classifier
.. _`scikit-learn`: http://scikit-learn.org

Classification Tree
-------------------

.. autoclass:: TreeLearner
   :members:
