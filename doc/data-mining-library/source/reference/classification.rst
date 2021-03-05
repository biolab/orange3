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

The following code loads lenses dataset (four discrete attributes and discrete
class), constructs naive Bayesian learner, uses it on the entire dataset
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


.. index:: classification tree
   pair: classification; tree

Classification Tree
-------------------

Orange includes three implemenations of classification trees. `TreeLearner`
is home-grown and properly handles multinominal and missing values.
The one from scikit-learn, `SklTreeLearner`, is faster. Another home-grown,
`SimpleTreeLearner`, is simpler and still faster.

The following code loads iris dataset (four numeric attributes and discrete
class), constructs a decision tree learner, uses it on the entire dataset
to construct a classifier, and then prints the tree:

    >>> import Orange
    >>> iris = Orange.data.Table('iris')
    >>> tr = Orange.classification.TreeLearner()
    >>> classifier = tr(data)
    >>> printed_tree = classifier.print_tree()
    >>> for i in printed_tree.split('\n'):
    >>>     print(i)
    [50.  0.  0.] petal length ≤ 1.9
    [ 0. 50. 50.] petal length > 1.9
    [ 0. 49.  5.]     petal width ≤ 1.7
    [ 0. 47.  1.]         petal length ≤ 4.9
       [0. 2. 4.]         petal length > 4.9
       [0. 0. 3.]             petal width ≤ 1.5
       [0. 2. 1.]             petal width > 1.5
       [0. 2. 0.]                 sepal length ≤ 6.7
       [0. 0. 1.]                 sepal length > 6.7
    [ 0.  1. 45.]     petal width > 1.7

.. autoclass:: TreeLearner
   :members:

.. autoclass:: SklTreeLearner
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


.. index:: neural network
   pair: classification; neural network

Neural Network
--------------
.. autoclass:: NNClassificationLearner
   :members:


.. index:: Rule induction
   pair: classification; rules

CN2 Rule Induction
------------------

.. automodule:: Orange.classification.rules

.. autoclass:: CN2Learner
   :members:

.. autoclass:: CN2UnorderedLearner
   :members:

.. autoclass:: CN2SDLearner
   :members:

.. autoclass:: CN2SDUnorderedLearner
   :members:


Calibration and threshold optimization
--------------------------------------

.. automodule:: Orange.classification.calibration

.. autoclass:: ThresholdClassifier
   :members:

.. autoclass:: ThresholdLearner
   :members:

.. autoclass:: CalibratedClassifier
   :members:

.. autoclass:: CalibratedLearner
   :members:


Gradient Boosted Trees
----------------------

.. automodule:: Orange.classification.gb

.. autoclass:: GBClassifier
   :members:

.. automodule:: Orange.classification.catgb

.. autoclass:: CatGBClassifier
   :members:

.. automodule:: Orange.classification.xgb

.. autoclass:: XGBClassifier
   :members:

.. autoclass:: XGBRFClassifier
   :members:
