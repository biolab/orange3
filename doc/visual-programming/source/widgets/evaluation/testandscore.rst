Test & Score
============

Tests learning algorithms on data.

Inputs
    Data
        input dataset
    Test Data
        separate data for testing
    Learner
        learning algorithm(s)

Outputs
    Evaluation Results
        results of testing classification algorithms


The widget tests learning algorithms. Different sampling schemes are
available, including using separate test data. The widget does two
things. First, it shows a table with different classifier performance
measures, such as `classification accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_ 
and `area under the curve <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.
Second, it outputs evaluation results, which can be used by other
widgets for analyzing the performance of classifiers, such as :doc:`ROC Analysis <../evaluation/rocanalysis>` or :doc:`Confusion Matrix <../evaluation/confusionmatrix>`.

The *Learner* signal has an uncommon property: it can be connected to more
than one widget to test multiple learners with the same procedures.

.. figure:: images/TestLearners-stamped.png

1. The widget supports various sampling methods.

   -  `Cross-validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_
      splits the data into a given number of folds (usually 5 or 10). The
      algorithm is tested by holding out examples from one fold at a time;
      the model is induced from other folds and examples from the held out
      fold are classified. This is repeated for all the folds.
   -  **Leave-one-out** is similar, but it holds out one instance at a
      time, inducing the model from all others and then classifying the
      held out instances. This method is obviously very stable, reliable …
      and very slow.
   -  **Random sampling** randomly splits the data into the training and
      testing set in the given proportion (e.g. 70:30); the whole procedure
      is repeated for a specified number of times.
   -  **Test on train data** uses the whole dataset for training and then
      for testing. This method practically always gives wrong results.
   -  **Test on test data**: the above methods use the data from *Data*
      signal only. To input another dataset with testing examples (for
      instance from another file or some data selected in another widget),
      we select *Separate Test Data* signal in the communication channel
      and select Test on test data.

2. For classification, *Target class* can be selected at the bottom of the widget.
   When *Target class* is (Average over classes),
   methods return scores that are weighted averages over all classes.
   For example, in case of the classifier with 3 classes,
   scores are computed for class 1 as a target class, class 2 as a target class,
   and class 3 as a target class. Those scores are averaged with weights
   based on the class size to retrieve the final score.

3. Produce a report. 
4. The widget will compute a number of performance statistics:

Classification
--------------

.. figure:: images/TestLearners.png

-  `Area under ROC <http://gim.unmc.edu/dxtests/roc3.htm>`_ is the
   area under the receiver-operating curve.
-  `Classification accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_
   is the proportion of correctly classified examples.
-  `F-1 <https://en.wikipedia.org/wiki/F1_score>`_ is a weighted harmonic mean of precision and recall (see below).
-  `Precision <https://en.wikipedia.org/wiki/Precision_and_recall>`_ is the proportion of true positives among instances classified as positive, e.g. the proportion of *Iris virginica* correctly identified as Iris virginica.
-  `Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_ is the proportion of true positives among all positive instances in
   the data, e.g. the number of sick among all diagnosed as sick.

Regression
----------

.. figure:: images/TestLearners-regression.png

- `MSE <https://en.wikipedia.org/wiki/Mean_squared_error>`_ measures the average of the squares of the errors or deviations (the difference between the estimator and what is estimated).
- `RMSE <https://en.wikipedia.org/wiki/Root_mean_square>`_ is the square root of the arithmetic mean of the squares of a set of numbers (a measure of imperfection of the fit of the estimator to the data)
- `MAE <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ is used to measure how close forecasts or predictions are to eventual outcomes. 
- `R2 <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_ is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable. 

Example
-------

In a typical use of the widget, we give it a dataset and a few learning
algorithms and we observe their performance in the table inside the
**Test & Score** widget and in the :doc:`ROC <../evaluation/rocanalysis>`. The data is often
preprocessed before testing; in this case we did some manual feature
selection (:doc:`Select Columns <../data/selectcolumns>` widget) on *Titanic* dataset, where we
want to know only the sex and status of the survived and omit the age.

.. figure:: images/TestLearners-example-classification.png

Another example of using this widget is presented in the documentation
for the :doc:`Confusion Matrix <../evaluation/confusionmatrix>` widget.

Errors, Warnings, Information
-----------------------------

Errors
++++++

*Train dataset is empty* - Your data input is present, but it does not contain any data instances. Check your data in a Data Table.
*Test dataset is empty* - Your test data is present, but it does not contain any data instances. Check your test data in a Data Table.
*Train data input requires a target variable* - Your data input is present, but the target variable isn't defined. Use :doc:`Select Columns <../data/selectcolumns>` to define the target variable.
*Too many target variables* - Your data has more than one target variable. Orange currently doesn't support multilabel prediction. Use :doc:`Select Columns <../data/selectcolumns>` to select a single target variable.
*Test data input requires a target variable* - Your test data input is present, but the target variable isn't defined. Use :doc:`Select Columns <../data/selectcolumns>` to define the target variable.
*Number of folds exceeds the data size* - Your train data has less data instances than the number of folds specified. Please decrease the number of folds, or, better, select the *Leave one out* option.
*Test and train datasets have different target variables* - Target variables from train and test data do not match. Use :doc:`Select Columns <../data/selectcolumns>` to define the same target variable for both data inputs.
*Not enough memory* - Your computer's memory cannot handle the data of this size. Try sampling with :doc:`Data Sampler <../data/datasampler>` to decrease the size of your data.
*Target variable has no values* - Target variable in the data has only missing values meaning the model can't be constructed. Use :doc:`Select Columns <../data/selectcolumns>` to select a different target variable.
*Target variable has only one value* - Target variable in the data has a constant value meaning the model can't be constructed. Use :doc:`Select Columns <../data/selectcolumns>` to select a different target variable.

Warnings
++++++++

"Instances with unknown target values were removed from data." - If a data instance has missing values, it was not considered in the computation. To change this, use :doc:`Impute <../data/impute>` to impute the the missing values.
"Missing separate test data input." - To Test on test data, use Test Data input. Double-click on the connection between the data and Test & Score and rewire the connection to Test Data.
"Some scores could not be computed." - The number of folds exceeds the number of instances in the smallest class. Decrease the number of folds to be the same or lower than the number of smallest class or use Leave one out option.
"Test data is present but unused. Select 'Test on test data' to use it." - Separate Test Data is present, but is not used. To see the results of testing on test data, select the final option, Test on test data.

Information
+++++++++++

"Train data has been sampled" - Train data is too big, so it has been randomly sampled for performance.
"Test data has been sampled" - Test data is too big, so it has been randomly sampled for performance.
