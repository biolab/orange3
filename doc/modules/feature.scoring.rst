.. py:currentmodule:: Orange.feature.scoring

#####################
Scoring (``scoring``)
#####################

.. index:: feature scoring

.. index::
   single: feature; feature scoring

Feature score is an assessment of the usefulness of the feature for
prediction of the dependant (class) variable. Orange provides classes
that compute the common feature scores for classification and regression
regression.

The code below computes the information gain of feature "tear_rate"
in the Lenses data set:

    >>> data = Orange.data.Table("lenses")
    >>> Orange.feature.scoring.InfoGain("tear_rate", data)
    0.54879494069539858

An alternative way of invoking the scorers is to construct the scoring
object, like in the following example:

    >>> gain = Orange.feature.scoring.InfoGain()
    >>> for att in data.domain.attributes:
    ...     print("%s %.3f" % (att.name, gain(att, data)))
    age 0.039
    prescription 0.040
    astigmatic 0.377
    tear_rate 0.549

==========================================
Feature scoring in classification problems
==========================================

.. index::
   single: feature scoring; information gain

.. class:: InfoGain

    Information gain is the expected decrease of entropy. See `Wikipedia entry on information gain
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

.. index::
   single: feature scoring; gain ratio

.. class:: GainRatio

    Information gain ratio is the ratio between information gain and
    the entropy of the feature's
    value distribution. The score was introduced in [Quinlan1986]_
    to alleviate overestimation for multi-valued features. See `Wikipedia entry on gain ratio
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

.. index::
   single: feature scoring; gini index

.. class:: Gini

    Gini index is the probability that two randomly chosen instances will have different
    classes. See `Wikipedia entry on gini index <http://en.wikipedia.org/wiki/Gini_coefficient>`_.

======================================
Feature scoring in regression problems
======================================

TBD.

.. rubric:: Bibliography

.. [Kononenko2007] Igor Kononenko, Matjaz Kukar: Machine Learning and Data Mining,
  Woodhead Publishing, 2007.

.. [Quinlan1986] J R Quinlan: Induction of Decision Trees, Machine Learning, 1986.

.. [Breiman1984] L Breiman et al: Classification and Regression Trees, Chapman and Hall, 1984.

.. [Kononenko1995] I Kononenko: On biases in estimating multi-valued attributes, International Joint Conference on Artificial Intelligence, 1995.
