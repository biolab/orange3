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
and its modus. The model is represented as an instance of
:obj:`Orange.classification.ConstantModel`, which
classifies all instances to the majority class.

Accuracy of majority model serves as a baseline when evaluating other
classification models.


Example
=======

    >>> import Orange
    >>> data = Orange.data.Table('titanic')
    >>> majority = Orange.classification.MajorityLearner()
    >>> model = majority(data)
    >>> print(model)
    ConstantModel [ 0.67696502  0.32303498]
    >>> data[0].get_class()
    Value('survived', yes)
    >>> model(data[0])
    Value('survived', no)


MajorityLearner and ConstantModel
---------------------------------

.. autoclass:: MajorityLearner
   :members:

.. autoclass:: ConstantModel
   :members:
