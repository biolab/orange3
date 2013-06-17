.. py:currentmodule:: Orange.classification.naive_bayes

#######################
Naive Bayes (``naive_bayes``)
#######################

.. index:: Naive Bayes classifier

The module for `Naive Bayes`_ (NB) classification contains two implementations:

- :obj:`Orange.classification.naive_bayes.BayesClassifier` is based on the popular `scikit-learn`_ package.

- :obj:`Orange.classification.naive_bayes.BayesStorageClassifier` demonstrates the use of the contingency table provided by the underlying storage. It returns only the predicted value.

Example
=======

    >>> from Orange.data import Table
    >>> from Orange.classification.naive_bayes import *
    >>> iris = Table('iris')
    >>> fitter = NaiveBayes()
    >>> model = fitter(iris[:-1])
    >>> model(iris[-1])
    Value('iris', Iris-virginica)

.. _`Naive Bayes`: http://en.wikipedia.org/wiki/Naive_Bayes_classifier
.. _`scikit-learn`: http://scikit-learn.org