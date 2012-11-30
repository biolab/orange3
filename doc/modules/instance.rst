.. currentmodule:: Orange.data

Data instance
=============

Class :obj:`Instance` represents a data instance. The base class stores its
own data, and derived classes are used to represent views into various data
storages.

Like data tables, every data instance is associated with a domain and its
data is split into attributes, classes, meta attributes and the weight.

The instance's data can be retrieved through attributes :obj:x, :obj:y and
:obj:metas. For derived classes, changing this data does not necessarily
modify the corresponding data tables or other structures. Use indexing to
modify the data.

.. autoclass:: Instance
    :members:


Rows of Data Tables
-------------------

.. autoclass:: RowInstance