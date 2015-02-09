.. py:currentmodule:: Orange.classification.naive_bayes

#######################
Naive Bayes (``naive_bayes``)
#######################

.. index:: Naive Bayes classifier

The module for `Naive Bayes`_ (NB) classification contains two implementations:

- :obj:`Orange.classification.naive_bayes.BayesClassifier` is based on
  the `scikit-learn`_ package. It accepts only the data with discrete
  attributes. To consider continuous attributes as well, use
  discretization preprocessor.

- :obj:`Orange.classification.naive_bayes.BayesStorageClassifier` demonstrates
  the use of the contingency table provided by the underlying storage.
  It returns only the predicted value and not the associated probabilities.

Example
=======

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



.. _`Naive Bayes`: http://en.wikipedia.org/wiki/Naive_Bayes_classifier
.. _`scikit-learn`: http://scikit-learn.org