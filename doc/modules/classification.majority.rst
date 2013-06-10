.. py:currentmodule:: Orange.classification.majority

#######################
Majority (``majority``)
#######################

.. index:: majority classifier
   pair: classification; majority classifier

Accuracy of models is often compared with the "default accuracy",
that is, the accuracy of a model which classifies all instances
to the majority class. Fitting such a model consists of
computing the class distribution and its modus. The model is
represented as an instance of
:obj:`Orange.classification.majority.ConstantClassifier`.

Example
=======

    >>> iris = Orange.data.Table('iris')
    >>> maj = Orange.classification.majority.MajorityLearner()
    >>> model = maj(iris[30:110])
    >>> print(model)
    ConstantClassifier [ 0.25   0.625  0.125]
    >>> model(iris[0])
    Value('iris', Iris-versicolor)
