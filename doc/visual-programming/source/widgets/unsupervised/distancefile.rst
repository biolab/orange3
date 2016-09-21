Distance File
===============

.. figure:: icons/distance-file.png

Loads an existing distance file. 

Signals
-------

**Inputs**:

-  None

**Outputs**:

-  **Distance File**

   A distance matrix. 

Description
-----------

.. figure:: images/DistanceFile-stamped.png

1. Choose from a list of previously saved distance files.
2. Browse for saved distance files.
3. Reload the selected distance file. 
4. Information about the distance file (number of points, labelled/unlabelled)
5. Browse documentation data sets.
6. Produce a report. 

Example
-------

When you want to use a custom-set distance file that you've saved before, open the **Distance File** widget and select the desired file with the *Browse* icon. This widget loads the existing distance file. In the snapshot below, we loaded the transformed *Iris* distance matrix from the :doc:`Save Distance Matrix <../unsupervised/savedistancematrix>` example. We displayed the transformed data matrix in the :doc:`Distance Map <../unsupervised/distancemap>` widget. We also decided to display a distance map of the original *Iris* data set for comparison. 

.. figure:: images/DistanceFile-Example.png
