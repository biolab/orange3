.. currentmodule:: Orange.data

######################
Data Table (``table``)
######################

.. autoclass:: Orange.data.Table
    :members: columns

    Stores data instances as a set of 2d tables representing the independent
    variables (attributes, features) and dependent variables
    (classes, targets), and the corresponding weights and meta attributes.

    The data is stored in 2d numpy arrays :obj:`X`, :obj:`Y`, :obj:`W`,
    :obj:`metas`. The arrays may be dense or sparse. All arrays have the same
    number of rows. If certain data is missing, the corresponding array has
    zero columns.

    Arrays can be of any type; default is `float` (that is, double precision).
    Values of discrete variables are stored as whole numbers.
    Arrays for meta attributes usually contain instances of `object`.

    The table also stores the associated information about the variables
    as an instance of :obj:`Domain`. The number of columns must match the
    corresponding number of variables in the description.

    There are multiple ways to get values or entire rows of the table.

    - The index can be an int, e.g. `table[7]`; the corresponding row is
      returned as an instance of :obj:`RowInstance`.

    - The index can be a slice or a sequence of ints (e.g. `table[7:10]` or
      `table[[7, 42, 15]]`, indexing returns a new data table with the
      selected rows.

    - If there are two indices, where the first is an int (a row number) and
      the second can be interpreted as columns, e.g. `table[3, 5]` or
      `table[3, 'gender']` or `table[3, y]` (where `y` is an instance of
      :obj:`~Orange.data.Variable`), a single value is returned as an instance
      of :obj:`~Orange.data.Value`.

    - In all other cases, the first index should be a row index, a slice or
      a sequence, and the second index, which represent a set of columns,
      should be an int, a slice, a sequence or a numpy array. The result is
      a new table with a new domain.

    Rules for setting the data are as follows.

    - If there is a single index (an `int`, `slice`, or a sequence of row
      indices) and the value being set is a single scalar, all
      attributes (not including the classes) are set to that value. That
      is, `table[r] = v` is equivalent to `table.X[r] = v`.

    - If there is a single index and the value is a data instance
      (:obj:`Orange.data.Instance`), it is converted into the table's domain
      and set to the corresponding rows.

    - Final option for a single index is that the value is a sequence whose
      length equals the number of attributes and target variables. The
      corresponding rows are set; meta attributes are set to unknowns.

    - For two indices, the row can again be given as a single `int`, a
       `slice` or a sequence of indices. Column indices can be a single
       `int`, `str` or :obj:`Orange.data.Variable`, a sequence of them,
       a `slice` or any iterable. The value can be a single value, or a
       sequence of appropriate length.

    .. attribute:: domain

        Description of the variables corresponding to the table's columns.
        The domain is used for determining the variable types, printing the
        data in human-readable form, conversions between data tables and
        similar.

Constructors
------------

The preferred way to construct a table is to invoke a named constructor.

.. automethod:: Table.from_domain
.. automethod:: Table.from_table
.. automethod:: Table.from_table_rows
.. automethod:: Table.from_numpy
.. automethod:: Table.from_file

Inspection
----------

.. automethod:: Table.is_view
.. automethod:: Table.is_copy
.. automethod:: Table.ensure_copy
.. automethod:: Table.has_missing
.. automethod:: Table.has_missing_class
.. automethod:: Table.checksum

Row manipulation
----------------

.. automethod:: Table.append
.. automethod:: Table.extend
.. automethod:: Table.insert
.. automethod:: Table.clear
.. automethod:: Table.shuffle

Weights
-------

.. automethod:: Table.has_weights
.. automethod:: Table.set_weights
.. automethod:: Table.total_weight
