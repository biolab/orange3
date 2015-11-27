Regression
==========

.. index:: regression

Regression in Orange is from the interface very similar to classification. These both require class-labeled data. Just like in classification, regression is implemented with learners and regression models (regressors). Regression learners are objects that accept data and return regressors. Regression models are given data items to predict the value of continuous class:

.. literalinclude:: code/regression.py


Handful of Regressors
---------------------

.. index::
   single: regression; tree

Let us start with regression trees. Below is an example script that builds the tree from data on housing prices and prints out the tree in textual form:

.. literalinclude:: code/regression-tree.py
   :lines: 3-

The script outputs the tree::

   RM<=6.941: 19.9
   RM>6.941
   |    RM<=7.437
   |    |    CRIM>7.393: 14.4
   |    |    CRIM<=7.393
   |    |    |    DIS<=1.886: 45.7
   |    |    |    DIS>1.886: 32.7
   |    RM>7.437
   |    |    TAX<=534.500: 45.9
   |    |    TAX>534.500: 21.9

Following is initialization of few other regressors and their prediction of the first five data instances in housing price data set:

.. index::
   single: regression; linear

.. literalinclude:: code/regression-other.py
   :lines: 3-

Looks like the housing prices are not that hard to predict::

      y    linreg    rf ridge
      22.2   19.3  21.8  19.5
      31.6   33.2  26.5  33.2
      21.7   20.9  17.0  21.0
      10.2   16.9  14.3  16.8
      14.0   13.6  14.9  13.5


Cross Validation
----------------

Evaluation and scoring methods are available at ``Orange.evaluation``:

.. literalinclude:: code/regression-cv.py
   :lines: 3-

.. index:
   single: regression; root mean squared error

.. index:
   single: regression; R2

We have scored the regression two measures for goodnes of fit: `root-mean-square error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ and `coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_, or R squared. Random forest has the lowest root mean squared error::

      Learner  RMSE  R2
      linreg   4.88  0.72
      rf       4.70  0.74
      ridge    4.91  0.71
      mean     9.20 -0.00

Not much difference here. Each regression method has a set of parameters. We have been running them with default parameters, and parameter fitting would help. Also, we have included ``MeanLearner`` in a list of our regression; this regressors simply predicts the mean value from the training set, and is used as a baseline.