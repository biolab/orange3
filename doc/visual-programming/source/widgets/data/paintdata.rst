Paint Data
==========

.. figure:: icons/paint-data.png

Paints the data on a 2D plane. You can place individual data points or
use a brush to paint larger data sets.

Signals
-------

**Inputs**:

-  (None)

**Outputs**:

-  **Data**

Attribute-valued data set created in the widget.

Description
-----------

The widget supports the creation of the new data set by visually placing
data points on a two-dimension plane. Data points can be placed on the
plane individually (*Put*) or in a larger number by brushing (*Brush*).
Data points can belong to classes if the data is intended to be used in
supervised learning.

.. figure:: images/PaintData-stamped.png

1. Name the axes and selet the class to paint data instances. You can
   add or remove classes. Use one class only to create classless,
   unsupervised data sets.
2. Drawing tools. Paint data points with *Brush* (multiple data
   instances) or *Put* (individual data instance). Select data points
   with *Select* and remove them with Delete/Backspace key. Reposition
   data points with `Jitter <https://en.wikipedia.org/wiki/Jitter>`__
   (spread) and *Magnet* (focus). Use *Zoom* and scroll to zoom in or
   out. Below, set radius and intensity for Brush, Put, Jitter and
   Magnet tools.
3. Tick the box on the left to automacitally commit changes to other
   widgets. Alternatively, press '*Send*' to apply them.
4. *Save graph* saves the graph to the computer in a .svg or .png
   format.

Example
-------

In the example below we have painted data with 4 classes. Such data set
is great for demonstrating k-means and hierarchical clustering methods.
In the screenshot we see that k-means, overall, recognizes clusters
better than hierarchical clustering. It returns a score rank, where the
best score (the one with the highest value) means the most likely number
of clusters. Hierarchical clustering, however, doesn’t group the right
classes together. This is a great tool for learning and exploring
statistical concepts.

.. figure:: images/PaintData-Example.png
