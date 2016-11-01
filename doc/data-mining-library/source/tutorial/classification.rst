Classification
==============

.. index:: classification
.. index::
   single: data mining; supervised

Much of Orange is devoted to machine learning methods for classification, or supervised data mining. These methods rely on the data with class-labeled instances, like that of senate voting. Here is a code that loads this data set, displays the first data instance and shows its predicted class (``republican``)::

   >>> data = Orange.data.Table("voting")
   >>> data[0]
   [n, y, n, y, y, ... | republican]

Orange implements function for construction of classification models, their evaluation and scoring. In a nutshel, here is the code that reports on cross-validated accuracy and AUC for logistic regression and random forests:

.. literalinclude:: code/classification-cv3.py

It turns out that for this domain logistic regression does well::

    Accuracy: [ 0.96321839  0.95632184]
    AUC: [ 0.96233796  0.95671252]

For supervised learning, Orange uses learners. These are objects that recieve the data and return classifiers. Learners are passed to evaluation rutines, such as cross-validation above.

Learners and Classifiers
------------------------

.. index::
   single: classification; learner
.. index::
   single: classification; classifier
.. index::
   single: classification; logistic regression

Classification uses two types of objects: learners and classifiers. Learners consider class-labeled data and return a classifier. Given the first three data instances, classifiers return the indexes of predicted class::

    >>> import Orange
    >>> data = Orange.data.Table("voting")
    >>> learner = Orange.classification.LogisticRegressionLearner()
    >>> classifier = learner(data)
    >>> classifier(data[:3])
    array([ 0.,  0.,  1.])

Above, we read the data, constructed a logistic regression learner, gave it the data set to construct a classifier, and used it to predict the class of the first three data instances. We also use these concepts in the following code that predicts the classes of the selected three instances in the data set:

..  literalinclude:: code/classification-classifier1.py
    :lines: 4-

The script outputs::

    democrat, originally democrat
    republican, originally democrat
    republican, originally republican

Logistic regression has made a mistake in the second case, but otherwise predicted correctly. No wonder, since this was also the data it trained from. The following code counts the number of such mistakes in the entire data set:

..  literalinclude:: code/classification-accuracy-train.py
    :lines: 4-

Probabilistic Classification
----------------------------

To find out what is the probability that the classifier assigns to, say, democrat class, we need to call the classifier with additional parameter that specifies the classification output type.

..  literalinclude:: code/classification-classifier2.py
    :lines: 3-

The output of the script also shows how badly the logistic regression missed the class in the second case::

    Probabilities for democrat:
    0.999506847581 democrat
    0.201139534658 democrat
    0.042347504805 republican

Cross-Validation
----------------

.. index:: cross-validation

Validating the accuracy of classifiers on the training data, as we did above, serves demonstration purposes only. Any performance measure that assess accuracy should be estimated on the independent test set. Such is also a procedure called `cross-validation <http://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_, which averages the evaluation scores across several runs, each time considering a different training and test subsets as sampled from the original data set:

.. literalinclude:: code/classification-cv.py
   :lines: 3-

.. index::
   single: classification; scoring
.. index::
   single: classification; area under ROC
.. index::
   single: classification; accuracy

Cross-validation is expecting a list of learners. The performance estimators also return a list of scores, one for every learner. There was just one learner (`lr`) in the script above, hence the array of length one was return. The script estimates classification accuracy and area under ROC curve::

    Accuracy: 0.779
    AUC:      0.704


Handful of Classifiers
----------------------

Orange includes a variety of classification algorithms, most of them wrapped from `scikit-learn <http://scikit-learn.org>`_, including:

- logistic regression (``Orange.classification.LogisticRegressionLearner``)
- k-nearest neighbors (``Orange.classification.knn.KNNLearner``)
- support vector machines (say, ``Orange.classification.svm.LinearSVMLearner``)
- classification trees (``Orange.classification.tree.SklTreeLearner``)
- radnom forest (``Orange.classification.RandomForestLearner``)

Some of these are included in the code that estimates the probability of a target class on a testing data. This time, training and test data sets are disjoint:

.. index::
   single: classification; logistic regression
.. index::
   single: classification; trees
.. index::
   single: classification; k-nearest neighbors

.. literalinclude:: code/classification-other.py

For these five data items, there are no major differences between predictions of observed classification algorithms::

    Probabilities for republican:
    original class  tree  knn   logreg
    republican      0.991 1.000 0.966
    republican      0.991 1.000 0.985
    democrat        0.000 0.000 0.021
    republican      0.991 1.000 0.979
    republican      0.991 0.667 0.963

The following code cross-validates these learners on the titanic data set.

.. literalinclude:: code/classification-cv2.py

Logistic regression wins in area under ROC curve::

             tree knn  logreg
    Accuracy 0.79 0.47 0.78
    AUC      0.68 0.56 0.70
