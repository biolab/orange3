###################################
Classification (``classification``)
###################################

.. automodule:: Orange.classification


.. index:: logistic regression
   pair: classification; logistic regression

Logistic Regression
-------------------
.. autoclass:: LogisticRegressionLearner
   :members:



.. index:: random forest
   pair: classification; random forest

Random Forest
-------------
.. autoclass:: RandomForestLearner
   :members:



.. index:: random forest (simple)
   pair: classification; simple random forest

Simple Random Forest
--------------------

.. autoclass:: SimpleRandomForestLearner
   :members:



.. index:: softmax regression classifier
   pair: classification; softmax regression

Softmax Regression
------------------

.. autoclass:: SoftmaxRegressionLearner
   :members:


.. index:: k-nearest neighbors classifier
   pair: classification; k-nearest neighbors

k-Nearest Neighbors
-------------------
.. autoclass:: KNNLearner
   :members:


.. index:: naive Bayes classifier
   pair: classification; naive Bayes

Naive Bayes
-----------
.. autoclass:: NaiveBayesLearner
   :members:

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


.. index:: multilayer perceptron
   pair: classification; multilayer perceptron

Multilayer perceptron (feed-forward neural network)
---------------------------------------------------
.. autoclass:: MLPLearner
   :members:


.. index:: SVM
   pair: classification; SVM

Support Vector Machines
-----------------------
.. autoclass:: SVMLearner
   :members:


.. index:: SVM, linear
   pair: classification; linear SVM

Linear Support Vector Machines
------------------------------
.. autoclass:: LinearSVMLearner
   :members:



.. index:: Nu-SVM
   pair: classification; Nu-SVM

Nu-Support Vector Machines
--------------------------
.. autoclass:: NuSVMLearner
   :members:



.. index:: one class SVM
   pair: classification; one class SVM

One Class Support Vector Machines
---------------------------------

.. autoclass:: OneClassSVMLearner
   :members:



.. index:: classification tree
   pair: classification; tree

Classification Tree
-------------------
.. autoclass:: TreeLearner
   :members:


.. index:: classification tree (simple)
   pair: classification; simple tree

Simple Tree
-----------
.. autoclass:: SimpleTreeLearner
   :members:



.. index:: majority classifier
   pair: classification; majority

Majority Classifier
-------------------

.. autoclass:: MajorityLearner
   :members:


.. index:: elliptic envelope
   pair: classification; elliptic envelope

Elliptic Envelope
-----------------

.. autoclass:: EllipticEnvelopeLearner
   :members:
