.. py:currentmodule:: Orange.evaluation.scoring

#############################
Scoring methods (``scoring``)
#############################

CA
--

.. index:: CA
.. autofunction:: Orange.evaluation.CA


Precision
---------

.. index:: Precision
.. autofunction:: Orange.evaluation.Precision


Recall
------

.. index:: Recall
.. autofunction:: Orange.evaluation.Recall


F1
--

.. index:: F1
.. autofunction:: Orange.evaluation.F1


PrecisionRecallFSupport
-----------------------

.. index:: PrecisionRecallFSupport
.. autofunction:: Orange.evaluation.PrecisionRecallFSupport


AUC
--------

.. index:: AUC
.. autofunction:: Orange.evaluation.AUC


Log Loss
--------

.. index:: Log loss
.. autofunction:: Orange.evaluation.LogLoss


MSE
---

.. index:: MSE
.. autofunction:: Orange.evaluation.MSE


MAE
---

.. index:: MAE
.. autofunction:: Orange.evaluation.MAE


R2
--

.. index:: R2
.. autofunction:: Orange.evaluation.R2


CD diagram
----------

.. index:: CD diagram

.. autofunction:: Orange.evaluation.compute_CD
.. autofunction:: Orange.evaluation.graph_ranks

Example
=======

    >>> import Orange
    >>> import matplotlib.pyplot as plt
    >>> names = ["first", "third", "second", "fourth" ]
    >>> avranks =  [1.9, 3.2, 2.8, 3.3 ]
    >>> cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
    >>> Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    >>> plt.show()

The code produces the following graph:

.. image:: images/statExamples-graph_ranks1.png
