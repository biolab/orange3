Rank
====

![image](icons/rank.png)

Ranking of attributes in classification or regression data sets.

Signals
-------

**Inputs**:

- **Data**

  Input data set.

**Outputs**:

- **Reduced Data**

  Data set whith selected attributes.

Description
-----------

**Rank** widget considers class-labeled data sets (classification or
regression) and scores the attributes according to their correlation
with the class.

![image](images/Rank-stamped.png)

1. Select attributes from the data table.
2. Data table with attributes (rows) and their scores by different scoring methods (columns)
3. If '*Commit on any change*' ticked, the widget automatically communicates changes to other widgets.

Example: Attribute Ranking and Selection
----------------------------------------

Below we have used the **Rank** widget immediately after the **File** widget to reduce the set
of data attributes and include only the most informative ones:

![image](images/Rank-Select-Schema.png)

Notice how the widget outputs a data set that includes only the
best-scored attributes:

![image](images/Rank-Select-Widgets.png)

Example: Feature Subset Selection for Machine Learning
------------------------------------------------------

What follows is a bit more complicated example. In the workflow below we
first split the data into training and test set. In the upper branch the
training data passes through the **Rank** widget to select the most
informative attributes, while in the lower branch there is no feature
selection. Both feature selected and original data sets are passed to
their own **Test Learners** widgets, which develop a
*Naive Bayes* classifier and score it on a test set.

<img src="images/Rank-and-Test.png" alt="image" width="400">

For data sets with many features a naive Bayesian classifier feature
selection, as shown above, would often yield a better predictive
accuracy.
