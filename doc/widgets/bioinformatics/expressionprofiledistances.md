Expression Profile Distances
============================

![image](icons/expression-profile-distances.png)

Computes distances between gene expression levels by groups.

Signals
-------

**Inputs**:

- **Data**

  Data set.

**Outputs**:

- **Distances**

  Distance matrix.
  
- **Sorted Data**

  Data with groups as attributes.

Description
-----------

Widget **Expression Profile Distances** computes distances between expression levels among instance groups.
Groups are data clusters set by the user through *separate by* function in the widget. Data can be separated by one or
more labels (usually timepoint, replicates, IDs, etc.). Widget outputs distance matrix that can be fed into
**Distance Map** and **Hierarchical Clustering** widgets.

![Distances Widget](images/Distances-stamped.png)

1. Choose which distances to measure, between rows or columns.
2. Choose the *Distance Metric*:
    - [**Euclidean**](https://en.wikipedia.org/wiki/Euclidean_distance) ("straight line", distance between two points)
    - [**Spearman**](https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient) (linear correlation between the rank of the values, remapped as a distance in a [0, 1] interval)
    - [**Pearson**](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) (linear correlation between the values, remapped as a distance in a [0, 1] interval)
3. Tick '*Apply on any change*' to automatically commit changes to other widgets. Alternatively, press '*Apply*'.

Example
-------

for instance to **Distance Map** to visualize distances, **Hierarchical Clustering** to cluster the attributes.

<img src="images/DistancesExample.png" alt="image" width="600">
