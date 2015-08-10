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

![Outliers]

Examples
--------

Below is a simple example how to use this widget. The input is fed
directly from the File widget, and the output Examples with Z-score to
the Data Table widget.

![Schema with Outliers]
