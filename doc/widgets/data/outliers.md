Outliers
========

![image](icons/outliers.png)

Simple outlier detection by comparing distances between instances.

Signals
-------

**Inputs**:

- **Data**

  Data set.

- **Distances**

  Distance matrix.

**Outputs**:

- **Outliers**

  Data set containing instances scored as outliers.

- **Inliers**

  Data set containing instances not scored as outliers.

Description
-----------

**Outliers** widget first computes distances between each pair of instances
in input Examples. Average distance between example to its nearest
examples is valued by a Z-score. Z-scores higher than zero denote an
example that is more distant to other examples than average. Input can
also be a distance matrix: in this case precalculated distances are
used.

Two parameters for Z-score calculation can be choosen: distance metrics
and number of nearest examples to which example’s average distance is
computed. Also, minimum Z-score to consider an example as outlier can be
set. Note, that higher the example’s Z-score, more distant is the
example from other examples.

Changes are applied automatically.

![Outliers](images/Outliers1-stamped.png)

1. Information on the input data, number of inliers and outliers based on the selected model.
2. Select the *Outlier detection method*:
   - **One class SVM with non-linear kernel (RBF)**: classifies data as similar or different from the core class
     - **Nu** is a parameter for the upper bound on the fraction of training errors and a lower 
       bound of the fraction of support vectors
     - **Kernel coefficient** is the gamma parameter, which specifies how much influence a single data instance has
   - **Covariance estimator**: fits ellipsis to central points with Mahalanobis distance metric
     - **Contamination** is the proportion of outliers in the data set
     - **Support fraction** specifies the proportion of points included in the estimate
3. Click *Detect outliers* to output the data.

Example
-------

Below is a simple example of how to use this widget. We use the *Iris* data set
to detect the outliers. We chose the *one class SVM with non-linear kernel (RBF)* method,
with Nu set at 20% (less training errors, more support vectors). Then we observe the outliers
in the **Data Table** widget, while we send the inliers to the **Scatter Plot**.

<img src="images/Outliers-Example.png" alt="image" width="600">
