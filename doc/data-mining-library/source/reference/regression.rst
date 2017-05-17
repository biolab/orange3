###########################
Regression (``regression``)
###########################

.. automodule:: Orange.regression


.. index:: .. index:: linear fitter
   pair: regression; linear fitter

Linear Regression
-----------------

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



.. index:: mean fitter
   pair: regression; mean fitter


Polynomial
----------

*Polynomial model* is a wrapper that constructs polynomial features of
a specified degree and learns a model on them.

.. autoclass:: Orange.regression.linear.PolynomialLearner


Mean
----

*Mean model* predicts the same value (usually the distribution mean) for all
data instances. Its accuracy can serve as a baseline for other regression
models.

The model learner (:class:`MeanLearner`) computes the mean of the given data or
distribution. The model is stored as an instance of :class:`MeanModel`.

Example
=======

    >>> from Orange.data import Table
    >>> from Orange.regression import MeanLearner
    >>> data = Table('auto-mpg')
    >>> learner = MeanLearner()
    >>> model = learner(data)
    >>> print(model)
    MeanModel(23.51457286432161)
    >>> model(data[:4])
    array([ 23.51457286,  23.51457286,  23.51457286,  23.51457286])

.. autoclass:: MeanLearner
   :members:



.. index:: random forest
   pair: regression; random forest

Random Forest
-------------
.. autoclass:: RandomForestRegressionLearner
   :members:



.. index:: random forest (simple)
   pair: regression; simple random forest

Simple Random Forest
--------------------

.. autoclass:: SimpleRandomForestLearner
   :members:



.. index:: regression tree
   pair: regression; tree

Regression Tree
-------------------

Orange includes two implemenations of regression tres: a home-grown one, and one
from scikit-learn. The former properly handles multinominal and missing values,
and the latter is faster.

.. autoclass:: TreeLearner
   :members:

.. autoclass:: SklTreeRegressionLearner
   :members:


.. index:: neural network
   pair: regression; neural network

Neural Network
--------------
.. autoclass:: NNRegressionLearner
   :members:
