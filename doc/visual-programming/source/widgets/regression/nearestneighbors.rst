Nearest Neighbors
=================

.. figure:: icons/k-nearest-neighbors-regression.png

Predicts according to the nearest training instances.

Signals
-------

**Inputs**:

-  **Data**

   A data set

-  **Preprocessor**

   Preprocessed data

**Outputs**:

-  **Learner**

   A learning algorithm with supplied parameters

-  **Predictor**

   A trained regressor. Signal *Predictor* sends the output signal only if
   input *Data* is present.

Description
-----------

The **Nearest Neighbors** widget uses the `kNN algorithm <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_
that searches for k closest training examples in feature space and uses
their average as prediction.

.. figure:: images/NearestNeighbors-stamped.png

1. Learner/predictor name
2. Set the number of nearest neighbors and the distance parameter
   (metric) as regression criteria. Metric can be:

   -  `Euclidean <https://en.wikipedia.org/wiki/Euclidean_distance>`_
      ("straight line", distance between two points)
   -  `Manhattan <https://en.wikipedia.org/wiki/Taxicab_geometry>`_
      (sum of absolute differences of all attributes)
   -  `Maximal <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
      (greatest of absolute differences between attributes)
   -  `Mahalanobis <https://en.wikipedia.org/wiki/Mahalanobis_distance>`_
      (distance between point and distribution).

3. You can assign weight to the contributions of the neighbors. The *Weights* you can use are:

   -  **Uniform**: all points in each neighborhood are weighted equally. 
   -  **Distance**: closer neighbors of a query point have a greater influence than the neighbors further away.

4. Produce a report. 
5. Press *Apply* to commit changes.

Example
-------

Below, is a workflow showing how to use both the *Predictor* and the
*Learner* output. For the purpose of this example, we used the *Housing* data set. For the *Predictor*, we input the prediction model into the
:doc:`Predictions <../evaluation/predictions>` widget and view the results in the :doc:`Data Table <../data/datatable>`. For
*Learner*, we can compare different learners in the :doc:`Test&Score <../evaluation/testlearners>` widget.

.. figure:: images/NearestNeighbors-example.png
