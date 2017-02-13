Randomize
=========

.. figure:: icons/randomize.png

Shuffles classes, attributes and/or metas of an input data set.

Signals
-------

**Inputs**:

-  **Data**

   Data set.

**Outputs**:

-  **Data**

   Randomized data set.

Description
-----------

The **Randomize** widget receives a data set in the input and outputs the same
data set in which the classes, attributes or/and metas are shuffled.

.. figure:: images/Randomize-Default.png

1. Select group of columns of the data set you want to shuffle.

2. Select proportion of the data set you want to shuffle.

3. Produce replicable output.

4. If *Apply automatically* is ticked, changes are committed automatically.
   Otherwise, you have to press *Apply* after each change.
5. Produce a report.

Example
-------

The **Randomize** widget is usually placed right after
(e.g. :doc:`File <../data/file>` widget. The basic usage is shown in the following
workflow, where values of class variable of Iris data set are randomly shuffled.

.. figure:: images/Randomize-Example1.png

In the next example we show how shuffling class values influences model
performance on the same data set as above.

.. figure:: images/Randomize-Example2.png
