Nomogram
========

.. figure:: icons/nomogram.png

Nomograms for visualization of Naive Bayes and Logistic Regression classifiers.

Signals
-------

**Inputs**:

-  **Classifier**

   A trained classifier (Naive Bayes or Logistic regression).

-  **Data**

   Data instance.

Description
-----------

**Nomogram** widget models the probability of the selected class. It also reveals the structure
of the model and the relative influences of the feature values to the class probabilities.
The probability for the chosen class is computed by 1. vs. all principle, which should be taken
in consideration when dealing with multiclass data (alternating probabilities do not sum to 1).
