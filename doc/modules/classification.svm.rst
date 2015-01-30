.. py:currentmodule:: Orange.classification.svm

#################################
Support Vector Machines (``svm``)
#################################

.. index:: support vector machines (SVM)
   pair: classification; support vector machines (SVM)

The module for `Support Vector Machine`_ (SVM) classification is based
on the popular `scikit-learn`_ package, which uses `LibSVM`_ and `LIBLINEAR`_
libraries. It provides several learning algorithms:

- :obj:`Orange.classification.svm.SVMLearner`, a general SVM learner;
- :obj:`Orange.classification.svm.LinearSVMLearner`, a fast learner useful for
    data sets with a large number of features;
- :obj:`Orange.classification.svm.NuSVMLearner`, a general SVM learner that uses
    a parameter to control the number of support vectors;
- :obj:`Orange.classification.svm.SVRLearner`, a general learner for support
    vector regression;
- :obj:`Orange.classification.svm.NuSVRLearner`, which is similar to
    `NuSVMLearner` for regression;
- :obj:`Orange.classification.svm.OneClassSVMLearner`, unsupervised learner
    useful for novelty and outlier detection.

Example
=======

    >>> from Orange import data
    >>> from Orange.classification import svm
    >>> from Orange.evaluation import testing
    >>> from Orange.evaluation.scoring import CA
    >>> iris = data.Table('iris')
    >>> results = testing.CrossValidation(iris, [svm.SVMLearner()], k=10)
    >>> CA(results)
    0.9866


.. _`Support Vector Machine`: http://en.wikipedia.org/wiki/Support_vector_machine
.. _`LibSVM`: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
.. _`LIBLINEAR`: http://www.csie.ntu.edu.tw/~cjlin/liblinear/
.. _`scikit-learn`: http://scikit-learn.org