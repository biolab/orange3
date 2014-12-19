.. py:currentmodule:: Orange.regression.mean

###############
Mean (``mean``)
###############

.. index:: mean fitter
   pair: regression; mean fitter

Fitting a mean model consists of computing mean value of class
distribution. The regression model is represented as an instance of
:obj:`Orange.regression.mean.MeanModel`, which returns the
mean value for all instances.

Accuracy of mean model servers as a baseline when evaluating other
regression models.


Example
=======

    >>> from Orange.regression.mean import MeanFitter
    >>> mpg = Orange.data.Table('auto-mpg')
    >>> mean_ = MeanFitter()
    >>> model = mean_(mpg[30:110])
    >>> print(model)
    MeanModel 18.4625
    >>> mpg[1].get_class()
    Value('mpg', 15.0))
    >>> model(mpg[1])
    Value('mpg', 18.5),

MeanFitter and MeanModel
------------------------

.. autoclass:: MeanFitter
   :members:

.. autoclass:: MeanModel
   :members:
