.. py:currentmodule:: Orange.preprocess

###################################
Data Preprocessing (``preprocess``)
###################################

.. index:: preprocessing

.. index::
   single: data; preprocessing

Preprocessing module contains data processing utilities like data
discretization, continuization, imputation and transformation.

Impute
======


.. index:: discretize data
   single: feature; discretize

Discretize
==========

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


Data discretization uses feature discretization classes from :obj:`Orange.preprocess`
and applies them on entire data set.

Default discretization method (equal frequency with four intervals) can be replaced with other
discretization approaches as demonstrated below:

.. literalinclude:: code/discretization-table-method.py
    :lines: 3-5

Transformation procedure
------------------------

`Discretization Algorithms`_ return a discretized variable (with fixed
parameters) that can transform either the learning or the testing data.
Parameter learning is separate to the transformation, as in machine
learning only the training set should be used to induce parameters.

To obtain discretized features, call a discretization algorithm with
with the data and the feature to discretize. The feature can be given
either as an index name or :obj:`Orange.data.Variable`. The following
example creates a discretized feature::

    import Orange
    data = Orange.data.Table("iris.tab")
    disc = Orange.feature.discretization.EqualFreq(n=4)
    disc_var = disc(data, 0)

The values of the first attribute will be discretized the data is
transformed to the  :obj:`Orange.data.Domain` domain that includes
``disc_var``.  In the example below we add the discretized first attribute
to the original domain::

  ndomain = Orange.data.Domain([disc_var] + list(data.domain.attributes),
      data.domain.class_vars)
  ndata = Orange.data.Table(ndomain, data)
  print(ndata)

The printout::

  [[<5.150000, 5.1, 3.5, 1.4, 0.2 | Iris-setosa],
   [<5.150000, 4.9, 3.0, 1.4, 0.2 | Iris-setosa],
   [<5.150000, 4.7, 3.2, 1.3, 0.2 | Iris-setosa],
   [<5.150000, 4.6, 3.1, 1.5, 0.2 | Iris-setosa],
   [<5.150000, 5.0, 3.6, 1.4, 0.2 | Iris-setosa],
   ...
  ]

_`Discretization Algorithms`
----------------------------

.. autoclass:: EqualWidth

.. autoclass:: EqualFreq

.. autoclass:: EntropyMDL

Continuize
==========

.. include:: preprocess.continuize.rst

Score
=====

Preprocessors
=============
