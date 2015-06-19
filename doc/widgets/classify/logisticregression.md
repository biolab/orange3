Logistic Regression
===================

![image](icons/logistic-regression.png)

Logistic Regression Learner

Signals
-------

**Inputs**:

- **Data**

  Data set
- **Preprocessor**

  Preprocessed data

**Outputs**:

- **Learner**

  The logistic regression learning algorithm with settings as specified in the dialog.

- **Logistic Regression Classifier**

  Trained classifier (a subtype of Classifier). **Logistic Regression Classifier** sends data only if the data input is present.

Description
-----------

![Logistic Regression Widget](images/LogisticRegression-stamped.png)

1. A name under which the learner appears in other widgets. The default name is 'Logistic regression'.
2. The penalty type (either L1 or L2).
3. Regularization value (higher value means less regularization).
4. Set numerical tolerance (permitted deviation from expected value).

Example
-------

The widget is used just as any other widget for inducing a classifier. This is an example demonstrating the prediction value of logistic regress used on *voting.tab* data set. We first use a **Logistic Regression** learner to provide a LR classifier for **Predictions** widget. We want to see the quality of LR prediction model a person being a republic or a democrat based on their voting patterns. In **Select Attributes** we choose logistic regression as a feature and party as a class. Then we use **Scatterplot** to see which instances were correctly predicted and which were false.

<img src="images/LogisticRegression-example.png" alt="image" width="400">
