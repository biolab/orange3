.. currentmodule:: Orange.data

############################
Data Instance (``instance``)
############################

Class :obj:`Instance` represents a data instance, typically retrieved from a
:obj:`Orange.data.Table` or :obj:`Orange.data.sql.SqlTable`. The base class
contains a copy of the data; modifying does not change the data in the storage
from which the instance was retrieved. Derived classes
(e.g. :obj:`Orange.data.table.RowInstance`) can represent views into various
data storages, therefore changing them actually changes the data.

Like data tables, every data instance is associated with a domain and its
data is split into attributes, classes, meta attributes and the weight. Its
constructor thus requires a domain and, optionally, data. For the following
example, we borrow the domain from the Iris dataset. ::

    >>> from Orange.data import Table, Instance
    >>> iris = Table("iris")
    >>> inst = Instance(iris.domain, [5.2, 3.8, 1.4, 0.5, "Iris-virginica"])
    >>> inst
    [5.2, 3.8, 1.4, 0.5 | Iris-virginica]
    >>> inst0 = Instance(iris.domain)
    >>> inst0
    [?, ?, ?, ? | ?]

The instance's data can be retrieved through attributes :obj:`x`, :obj:`y` and
:obj:`metas`. ::

    >>> inst.x
    array([ 5.2,  3.8,  1.4,  0.5])
    >>> inst.y
    array([ 2.])
    >>> inst.metas
    array([], dtype=object)

Other utility functions provide for easier access to the instances data. ::

    >>> inst.get_class()
    Value('iris', Iris-virginica)
    >>> for e in inst.attributes():
    ...     print(e)
    ...
    5.2
    3.8
    1.4
    0.5

.. autoclass:: Instance
    :members:

    Constructor requires a domain and the data as numpy array, an existing
    instance from the same or another domain or any Python iterable.

    Domain can be omitted it the data is given as an existing data instances.

    When the instance is not from the given domain, Orange converts it.

        >>> from Orange.preprocess import DomainDiscretizer
        >>> discretizer = DomainDiscretizer()
        >>> d_iris = discretizer(iris)
        >>> d_inst = Instance(d_iris, inst)



Rows of Data Tables
-------------------

.. autoclass:: RowInstance
    :members:

    `RowInstance` is a specialization of :obj:`~Orange.data.Instance` that
    represents a row of :obj:`Orange.data.Table`. `RowInstance` is returned
    by indexing a `Table`.

    The difference between `Instance` and `RowInstance` is that the latter
    represents a view into the table: changing the `RowInstance` changes the
    data in the table::

        >>> iris[42]
        [4.4, 3.2, 1.3, 0.2 | Iris-setosa]
        >>> inst = iris[42]
        >>> inst.set_class("Iris-virginica")
        >>> iris[42]
        [4.4, 3.2, 1.3, 0.2 | Iris-virginica]

    Dense tables can also be modified directly through :obj:`x`, :obj:`y` and
    :obj:`metas`. ::

        >>> inst.x[0] = 5
        >>> iris[42]
        [5.0, 3.2, 1.3, 0.2 | Iris-virginica]

    Sparse tables cannot be changed in this way.
