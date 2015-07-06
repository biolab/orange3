Distances
=========

![image](icons/distances.png)

Computes distances between instances/attributes in the data set.

Signals
-------

**Inputs**:

- **Data**

  Data set

**Outputs**:

- **Distances**

  Distance matrix

Description
-----------

Widget Distances computes the distances between either the attributes
or instances in the data set.

![Association Rules Widget]

1. Choose the *Distance Metrics*:
    - [**Euclidean**](https://en.wikipedia.org/wiki/Euclidean_distance) ("straight line", distance between two points)
    - [**Manhattan**](https://en.wiktionary.org/wiki/Manhattan_distance) (the sum of absolute differences for all attributes)
    - [**Cosine**](https://en.wikipedia.org/wiki/Cosine_similarity) (the cosine of the angle between two vectors of an inner product space)
    - [**Jaccard**](https://en.wikipedia.org/wiki/Jaccard_index) (the size of the intersection divided by the size of the union of the sample sets)
    - [**Spearman**](https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient) (linear correlation between the rank of the values)
    - [**Spearman absolute**](https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient) (linear correlation between the rank of the absolute values)
    - [**Pearson**](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) (linear correlation between the values)
    - [**Pearson absolute**](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) (linear correlation between the absolute values)

In case of missing values the widget automatically imputes the average value of the row or the column.

Since the widget cannot compute distances between discrete and
continuous attributes, it only uses continuous attributes and ignores the discrete ones.
If you want to put the discrete attributes to use, continuize them
with **Continuize** widget first.

Examples
--------

This widget is an intermediate widget: it shows no user readable results
and its output needs to be fed to a widget that can do something useful
with the computed distances, for instance the Distance Map,
Hierarchical Clustering to cluster the attributes, or MDS to visualize
the distances between them.

![Association Rules]

  [image]: ../../../../Orange/OrangeWidgets/Unsupervised/icons/Distance.svg
  [Association Rules Widget]: images/AttributeDistance.png
  [attribute interactions]: http://stat.columbia.edu/~jakulin/Int/
  [Association Rules]: images/AttributeDistance-Schema.png
