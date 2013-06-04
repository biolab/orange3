.. py:currentmodule:: Orange.data.discretization

########################################
Data discretization (``discretization``)
########################################

.. index:: discretization

.. index::
   single: data; discretization

Continuous features in the data can be discretized using a uniform
discretization method. Discretizatione replaces continuous features with
the corresponding categorical features:

.. literalinclude:: code/discretization-table.py

Discretization introduces new categorical features with discretized values::

    Original data set:
    [5.1, 3.5, 1.4, 0.2 | Iris-setosa]
    [4.9, 3.0, 1.4, 0.2 | Iris-setosa]
    [4.7, 3.2, 1.3, 0.2 | Iris-setosa]
    Discretized data set:
    [<5.450000, >=3.150000, <2.450000, <0.800000 | Iris-setosa]
    [<5.450000, [2.850000, 3.150000), <2.450000, <0.800000 | Iris-setosa]
    [<5.450000, >=3.150000, <2.450000, <0.800000 | Iris-setosa]


Data discretization uses feature discretization classes from :obj:`Orange.feature.discretization`
and applies them on entire data set.

Default discretization method (equal frequency with four intervals) can be replaced with other
discretization approaches as demonstrated below:

.. literalinclude:: code/discretization-table-method.py
    :lines: 3-5

.. TODO write about removal of features if entropy-based discretization is used

Data discretization classes
===========================

.. autoclass:: DiscretizeTable
   :members:
   :special-members: __call__

