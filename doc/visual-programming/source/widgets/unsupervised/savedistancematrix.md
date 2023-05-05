Save Distance Matrix
====================

Saves a distance matrix.

If the file is saved to the same directory as the workflow or in the subtree of that directory, the widget remembers the relative path. Otherwise it will store an absolute path, but disable auto save for security reasons.

**Inputs**

- Distances: distance matrix

![](images/SaveDistanceMatrix-stamped.png)

1. By clicking *Save*, you choose from previously saved distance matrices. Alternatively, tick the box on the left side of the *Save* button and changes will be communicated automatically.
2. By clicking *Save as*, you save the distance matrix to your computer, you only need to enter the name of the file and click *Save*.

Example
-------

In the snapshot below, we used the [Distance Transformation](../unsupervised/distancetransformation.md) widget to transform the distances in the *Iris* dataset. We then chose to save the transformed version to our computer, so we could use it later on. We decided to output all data instances. You can choose to output just a minor subset of the data matrix. Pairs are marked automatically. If you wish to know what happened to our changed file, see [Distance File](../unsupervised/distancefile.md).

![](images/SaveDistanceMatrix-Example.png)
