Rank
====

Ranking of attributes in classification or regression datasets.

**Inputs**

- Data: input dataset
- Scorer: models for feature scoring

**Outputs**

- Reduced Data: dataset with selected attributes
- Scores: data table with feature scores
- Features: list of attributes

The **Rank** widget scores variables according to their correlation with discrete or numeric target variable, based on applicable internal scorers (like information gain, chi-square and linear regression) and any connected external models that supports scoring, such as linear regression, logistic regression, random forest, SGD, etc. The widget can also handle unsupervised data, but only by external scorers, such as PCA.

![](images/Rank-stamped.png)

1. Select scoring methods. See the options for classification, regression and unsupervised data in the **Scoring methods** section.
2. Select attributes to output. *None* won't output any attributes, while *All* will output all of them. With manual selection, select the attributes from the table on the right. *Best ranked* will output n best ranked attributes.
   If *Send Automatically* is ticked, the widget automatically communicates changes to other widgets.
3. Status bar. Produce a report by clicking on the file icon. Observe input and output of the widget. On the right, warnings and errors are shown.

Scoring methods (classification)
--------------------------------

1. Information Gain: the expected amount of information (reduction of entropy)
2. [Gain Ratio](https://en.wikipedia.org/wiki/Information_gain_ratio): a ratio of the information gain and the attribute's intrinsic information, which reduces the bias towards multivalued features that occurs in information gain
3. [Gini](https://en.wikipedia.org/wiki/Gini_coefficient): the inequality among values of a frequency distribution
4. [ANOVA](https://en.wikipedia.org/wiki/One-way_analysis_of_variance): the difference between average values of the feature in different classes
5. [Chi2](https://en.wikipedia.org/wiki/Chi-squared_distribution): dependence between the feature and the class as measured by the chi-square statistic
6. [ReliefF](https://en.wikipedia.org/wiki/Relief_(feature_selection)): the ability of an attribute to distinguish between classes on similar data instances
7. [FCBF (Fast Correlation Based Filter)](https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf): entropy-based measure, which also identifies redundancy due to pairwise correlations between features

Additionally, you can connect certain learners that enable scoring the features according to how important they are in models that the learners build (e.g. [Logistic Regression](../model/logisticregression.md), [Random Forest](../model/randomforest.md), [SGD](../model/stochasticgradient.md)). Please note that the data is normalized before ranking.

Scoring methods (regression)
----------------------------

1. [Univariate Regression](https://en.wikipedia.org/wiki/Simple_linear_regression): linear regression for a single variable
2. [RReliefF](http://www.clopinet.com/isabelle/Projects/reading/robnik97-icml.pdf): relative distance between the predicted (class) values of the two instances.

Additionally, you can connect regression learners (e.g. [Linear Regression](../model/linearregression.md), [Random Forest](../model/randomforest.md), [SGD](../model/stochasticgradient.md)). Please note that the data is normalized before ranking.

Scoring method (unsupervised)
-----------------------------

Currently, only [PCA](../unsupervised/PCA.md) is supported for unsupervised data. Connect PCA to Rank to obtain the scores. The scores correspond to the correlation of a variable with the individual principal component.

Scoring with learners
---------------------

Rank can also use certain learners for feature scoring. See [Learners as Scorers](../../learners-as-scorers/index.md) for an example.

Example: Attribute Ranking and Selection
----------------------------------------

Below, we have used the **Rank** widget immediately after the [File](../data/file.md) widget to reduce the set of data attributes and include only the most informative ones:

![](images/Rank-Select-Schema.png)

Notice how the widget outputs a dataset that includes only the best-scored attributes:

![](images/Rank-Select-Widgets.png)

Example: Feature Subset Selection for Machine Learning
------------------------------------------------------

What follows is a bit more complicated example. In the workflow below, we first split the data into a training set and a test set. In the upper branch, the training data passes through the **Rank** widget to select the most informative attributes, while in the lower branch there is no feature selection. Both feature selected and original datasets are passed to their own [Test & Score](../evaluate/testandscore.md) widgets, which develop a *Naive Bayes* classifier and score it on a test set.

![](images/Rank-and-Test.png)

For datasets with many features, a naive Bayesian classifier feature selection, as shown above, would often yield a better predictive accuracy.
