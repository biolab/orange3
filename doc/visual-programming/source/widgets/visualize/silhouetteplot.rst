Silhouette Plot
===============

.. figure:: icons/silhouette-plot.png

A graphical representation of consistency within clusters of data. 

Signals
-------

**Inputs**

-  **Data**

   A data set.

**Outputs**

-  **Selected Data**

   A subset of instances that the user has manually selected from the plot.

-  **Other Data**

   Remaining data.

Description
-----------

The **Silhouette Plot** widget offers a graphical representation of consistency within clusters of data and provides the user with the means to visually assess cluster quality. The silhouette score is a measure of how similar an object is to its own cluster in comparison to other clusters and is crucial in the creation of a silhoutte plot. The silhouette score close to 1 indicates that the data instance is close to the center of the cluster and instances posessing the silhouette scores close to 0 are on the border between two clusters. 

.. figure:: images/SilhouettePlot-stamped.png

1. Choose the distance metric. You can choose between:

   -  `Euclidean <https://en.wikipedia.org/wiki/Euclidean_distance>`_ ("straight line", distance between two points)
   -  `Manhattan <https://en.wiktionary.org/wiki/Manhattan_distance>`_ (the sum of absolute differences for all attributes)

2. Select the cluster label. You can decide whether to group the instances by cluster or not.
3. Display options:

   -  *Choose bar width*.
   -  *Annotations*: annotate the silhouette plot. 

4. *Save Image* saves the created silhouette plot to your computer in a *.png* or *.svg* format.
5. Produce a report.
6. Output:

   -  *Add silhouette scores* (good clusters have higher silhoutte scores)
   -  By clicking *Commit*, changes are comminicated to the output of the widget. Alternatively, tick the box on the left and changes will be communicated automatically. 

7. The created silhouette plot. 

Example
-------

In the snapshot below, we have decided to use the **Silhoutte Plot** on the *iris* data set. We selected data intances with low silhouette scores and passed them on as a subset to the :doc:`Scatter Plot <../visualize/scatterplot>` widget. This visualization only confirms the accuracy of the **Silhouette Plot** widget, as you can clearly see that the subset lies in the border between two clusters. 

.. figure:: images/SilhouettePlot-Example.png

If you are interested in other uses of the **Silhouette Plot** widget, feel free to explore our `blog post <http://blog.biolab.si/2016/03/23/all-i-see-is-silhouette/>`_.
