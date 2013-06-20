.. currentmodule:: Orange.data.storage

##########################
Data Storage (``storage``)
##########################

:obj:Orange.data.storage.Storage is an abstract class representing a data object
in which rows represent data instances (examples, in machine learning
terminology) and columns represent variables (features, attributes, classes,
targets, meta attributes).

Data is divided into three parts that represent independent variables (`X`),
dependent variables (`Y`) and meta data (`metas`). If practical, the class
should expose those parts as properties. In the associated domain
(:obj:`Orange.data.Domain`), the three parts correspond to lists of variable
descriptors `attributes`, `class_vars` and `metas`.

Any of those parts may be missing, dense, sparse or sparse boolean. The
difference between the later two is that the sparse data can be seen as a list
of pairs (variable, value), while in the latter the variable (item) is present
or absent, like in market basket analysis. The actual storage of sparse data
depends upon the storage type.

There is no uniform constructor signature: every derived class provides one or
more specific constructors.

There are currently two derived classes :obj:Orange.data.Table and
:obj:Orange.data.sql.Table, the former storing the data in-memory, in numpy
objects, and the latter in SQL (currently, only PostreSQL is supported).

Derived classes must implement at least the methods for getting rows and the
number of instances (`__getitem__` and `__len__`). To make storage fast enough
to be practically useful, it must also reimplement a number of filters,
preprocessors and aggregators. For instance, method
`_filter_values(self, filter)` returns a new storage which only contains the
rows that match the criteria given in the filter. :obj:Orange.data.Table
implements an efficient method based on numpy indexing, and
:obj:Orange.data.sql.Table, which "stores" a table as an SQL query, converts
the filter into a WHERE clause.

.. attribute:: domain (:obj:`Orange.data.Domain`)

    The domain describing the columns of the data


Data access
-----------

.. method:: __getitem__(self, index)

    Return one or more rows of the data.

    - If the index is an int, e.g. `data[7]`; the corresponding row is
      returned as an instance of :obj:~Orange.data.instance.Instance. Concrete
      implementations of `Storage` use specific derived classes for instances.

    - If the index is a slice or a sequence of ints (e.g. `data[7:10]` or
      `data[[7, 42, 15]]`, indexing returns a new storage with the selected
      rows.

    - If there are two indices, where the first is an int (a row number) and
      the second can be interpreted as columns, e.g. `data[3, 5]` or
      `data[3, 'gender']` or `data[3, y]` (where `y` is an instance of
      :obj:`~Orange.data.Variable`), a single value is returned as an instance
      of :obj:`~Orange.data.Value`.

    - In all other cases, the first index should be a row index, a slice or
      a sequence, and the second index, which represent a set of columns,
      should be an int, a slice, a sequence or a numpy array. The result is
      a new storage with a new domain.

.. method:: .__len__(self)

    Return the number of data instances (rows)


Inspection
----------

.. method:: Storage.X_density, Storage.Y_density, Storage.metas_density

    Indicates whether the attributes, classes and meta attributes are dense
    (`Storage.DENSE`) or sparse (`Storage.SPARSE`). If they are sparse and
    all values are 0 or 1, it is marked as (`Storage.SPARSE_BOOL`). The
    Storage class provides a default DENSE. If the data has no attibutes,
    classes or meta attributes, the corresponding method should re


Filters
-------

.. method:: _filter_is_defined(self, columns=None, negate=False)

    Extract rows without undefined values.

    :param columns: optional list of columns that are checked for unknowns
    :type columns: sequence of ints, variable names or descriptors
    :param negate: invert the selection
    :type negate: bool
    :return: a new storage of the same type or :obj:~Orange.data.Table
    :rtype: Orange.data.storage.Storage


.. method:: _filter_has_class(self, negate=False)

    Return rows with known value of the target attribute. If there are multiple
    classes, all must be defined.

    :param negate: invert the selection
    :type negate: bool
    :return: a new storage of the same type or :obj:~Orange.data.Table
    :rtype: Orange.data.storage.Storage


.. method:: _filter_same_value(self, column, value, negate=False)

    Select rows based on a value of the given variable.

    :param column: the column that is checked
    :type column: int, str or Orange.data.Variable
    :param value: the value of the variable
    :type value: int, float or str
    :param negate: invert the selection
    :type negate: bool
    :return: a new storage of the same type or :obj:~Orange.data.Table
    :rtype: Orange.data.storage.Storage


.. method:: _filter_values(self, filter)

    Apply a filter to the data.

    :param filter: A filter for selecting the rows
    :type filter: Orange.data.Filter
    :return: a new storage of the same type or :obj:~Orange.data.Table
    :rtype: Orange.data.storage.Storage

