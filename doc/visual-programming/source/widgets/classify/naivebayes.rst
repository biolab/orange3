Naive Bayes
===========

.. figure:: icons/naive-bayes.png

Naive Bayesian Learner

Signals
-------

**Inputs**:

-  **Data**

   A data set

-  **Preprocessor**

   Preprocessed data

**Outputs**:

-  **Learner**

   A `Naive Bayesian <https://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_
   learning algorithm with settings as specified in the dialog. It can be
   fed into widgets for testing learners.

-  **Naive Bayesian Classifier**

   A trained classifier (a subtype of Classifier). The *Naive Bayesian
   Classifier* signal sends data only if the learning data (signal
   **Data**) is present.

Description
-----------

.. figure:: images/NaiveBayes.png

This widget has two options: the name under which it will appear in
other widgets and producing a report. The default name is *Naive Bayes*. When you change it,
you need to press *Apply*.

Examples
--------

Here, we present two uses of this widget. First, we compare the results of the
**Naive Bayesian learner** with another learner, the :doc:`Random Forest <../classify/randomforest>`.

.. figure:: images/NaiveBayes-Predictions.png

The second schema shows the quality of predictions made with **Naive
Bayes**. We feed the :doc:`Test&Score <../evaluation/testlearners>` widget a Naive Bayes learner and
then send the data to the :doc:`Confusion Matrix <../evaluation/confusionmatrix>`. In this widget, we select the
misclassified instances and show them in :doc:`Scatterplot <../visualize/scatterplot>`. The bold dots
in the scatterplot are the misclassified instances from **Naive Bayes**.

.. figure:: images/NaiveBayes-Misclassifications.png
