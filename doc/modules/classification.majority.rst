.. py:currentmodule:: Orange.classification.majority

#######################
Majority (``majority``)
#######################

.. index:: majority classifier
   pair: classification; majority classifier

Accuracy of models is often compared with the "default accuracy",
that is, the accuracy of a model which classifies all instances
to the majority class.

Fitting a majority model consists of computing the class distribution
nd its modus. The model is represented as an instance of
:obj:`Orange.classification.majority.ConstantClassifier`, which
classifies all instances to the majority class.

Accuracy of majority model servers as a baseline when evaluating other
classification models.


Example
=======

    >>> from Orange.classificiation.majority import MajorityFitter
    >>> mpg = Orange.data.Table('auto-mpg')
    >>> majority = MajorityFitter()
    >>> model = majority(mpg[30:110])
    >>> print(model)
    ConstantClassifier [ 0.25   0.625  0.125]
    >>> iris[0].get_class()
    Value('iris', Iris-setosa)
    >>> model(iris[0])
    Value('iris', Iris-versicolor)


MajorityFitter and ConstantClassifier
-------------------------------------

.. autoclass:: MajorityFitter
   :members:

.. autoclass:: ConstantClassifier
   :members:
