Test & Score
============

![image](icons/test-learners.png)

Tests learning algorithms on data.

Signals
-------

**Inputs**:

- **Data**

  Data for training and, if there is no separate test data set, also testing.

- **Test Data**

  Separate data for testing.

- **Learner**

  One or more learning algorithms.

**Outputs**:

- **Evaluation results**

  Results of testing the algorithms.

Description
-----------

The widget tests learning algorithms. Different sampling schemes
are available, including using separate test data. The widget does two
things. First, it shows a table with different classifier performance measures,
such as [classification accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) and [area under ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).
Second, it outputs evaluation results, which can be used by other widgets
for analyzing the performance of classifiers, such as **ROC Analysis** or
**Confusion Matrix**.

*Learner* signal has an uncommon property: it can be
connected to more than one widgetto test multiple learners with the same procedures.

![image](images/TestLearners-stamped.png)

1. The widget supports various sampling methods.
  - [**Cross-validation**](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) splits the data into a given number of folds (usually 5 or 10).
    The algorithm is tested by holding out examples from one fold at a time; the model
    is induced from other folds and examples from the held out fold
    are classified. This is repeated for all the folds.
  - **Leave-one-out** is similar, but it holds out one instance
    at a time, inducing the model from all others and then classifying the 
    held out instances. This method is obviously very stable, reliable… and very
    slow.
  - **Random sampling** randomly splits the data into the training and
    testing set in the given proportion (e.g. 70:30); the whole procedure is
    repeated for a specified number of times.
  - **Test on train data** uses
    the whole data set for training and then for testing. This method
    practically always gives wrong results.
  - **Test on test data**: the above methods use the data from *Data* signal only. To input another
    data set with testing examples (for instance from another file or some
    data selected in another widget), we select *Separate
    Test Data* signal in the communication channel and select Test on test data.

2. Press *Apply* to apply the changes.

3. Only *Test on test data* requires a target class, e.g. having the disease or being of subvariety *Iris setosa*.
  When *Target class* is (None), methods return the average value. Target class can be selected at the bottom of the widget.

4. The widget will compute a number of performance statistics.
  - [**Area under ROC**](http://gim.unmc.edu/dxtests/roc3.htm) is the area under receiver-operating curve.
  - [**Classification accuracy**](https://en.wikipedia.org/wiki/Accuracy_and_precision) is the proportion of correctly classified examples.
  - [**F-1**](https://en.wikipedia.org/wiki/F1_score) is a weighted harmonic mean of precision and recall (see below).
  - [**Precision**](https://en.wikipedia.org/wiki/Precision_and_recall) is the proportion of true positives among instances classified as positive, e.g. the proportion of *Iris virginica* correctly identified as Iris virginica.
  - [**Recall**](https://en.wikipedia.org/wiki/Precision_and_recall) is the proportion of true positives among all positive instances in the data, e.g. the number of sick among all diagnosed as sick.

Example
-------

In a typical use of the widget, we give it a data set and a few learning
algorithms and we observe their performance in the table inside the
**Test & Score** widget and in the **ROC**. The data is often preprocessed before testing; in
this case we did some manual feature selection (**Select Columns** widget) on *Titanic* data set, where
we want to know only the sex and status of the survived and omit the age.

<img src="images/TestLearners-example.png" alt="image" width="600">

Another example of using this widget is presented in the documentation for **Confusion Matrix** widget.
