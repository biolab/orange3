.. py:currentmodule:: Orange.regression.linear

###################
Linear (``linear``)
###################

.. index:: linear fitter
   pair: regression; linear fitter


Linear regression is a statistical regression method which tries to
predict a value of a continuous response (class) variable based on
the values of several predictors. The model assumes that the response
variable is a linear combination of the predictors, the task of
linear regression is therefore to fit the unknown coefficients.


Example
=======

    >>> from Orange.regression.linear import LinearRegressionLearner
    >>> mpg = Orange.data.Table('auto-mpg')
    >>> mean_ = LinearRegressionLearner()
    >>> model = mean_(mpg[40:110])
    >>> print(model)
    LinearModel LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    >>> mpg[20]
    Value('mpg', 25.0)
    >>> model(mpg[0])
    Value('mpg', 24.6)

.. autoclass:: Orange.regression.linear.LinearRegressionLearner
.. autoclass:: Orange.regression.linear.RidgeRegressionLearner
.. autoclass:: Orange.regression.linear.LassoRegressionLearner
.. autoclass:: Orange.regression.linear.SGDRegressionLearner
.. autoclass:: Orange.regression.linear.LinearModel

