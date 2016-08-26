.. currentmodule:: Orange.data

######################
Data Table (``table``)
######################

.. autoclass:: Orange.data.Table
    :members: columns

    Stores data instances in a dense :obj:`pandas.DataFrame` representing the independent
    variables (attributes, features) and dependent variables
    (classes, targets), and the corresponding weights and meta attributes.

    2D numpy arrays :obj:`X`, :obj:`Y`, :obj:`W`, :obj:`metas` can be generated.
    The arrays may be dense or sparse. All arrays have the same
    number of rows. If certain data is missing, the corresponding array has
    zero columns.

    Arrays can be of any type; default is `float` (that is, double precision).
    Arrays for meta attributes usually contain instances of `object`.

    The table also stores the associated information about the variables
    as an instance of :obj:`Domain`. The number of columns must match the
    corresponding number of variables in the description.

    Indexing the table works the same as in `pandas`. In a nutshell

    - Use `table.iloc[i]` for position-based indexing.
    - Use `table.loc[i]` for index-based indexing.
    - Both accept tuples of `(row_index, column_name)` or slices where appropriate.
    - Use table[colname] to get columns.

    One-domensional alternatives are called `Series`.

    Setting data works the same as in `pandas`. The only thing you need to be careful
    of is that chaining indexers, as in `table.iloc[0].iloc[0]` won't work and
    shouldn't be used, use `table.iloc[0, 0]` instead.

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
.. automethod:: Table.from_dataframe
.. automethod:: Table.from_numpy
.. automethod:: Table.from_list
.. automethod:: Table.from_file
.. automethod:: Table.from_url

Getting Data
------------
.. automethod:: Table.X
.. automethod:: Table.Y
.. automethod:: Table.metas
.. automethod:: Table.weights

Inspection
----------

.. automethod:: Table.has_weights
.. automethod:: Table.approx_len
.. automethod:: Table.exact_len
.. automethod:: Table.has_missing
.. automethod:: Table.density
.. automethod:: Table.is_dense
.. automethod:: Table.is_sparse

Manipulation
------------

.. automethod:: Table.concatenate
.. automethod:: Table.merge

Weights
-------

.. automethod:: Table.weights
.. automethod:: Table.set_weights

Aggregators
-----------

Similarly to filters, storage classes should provide several methods for fast
computation of statistics. These methods are not called directly but by modules
within :obj:`Orange.statistics`.

.. method:: _compute_basic_stats(
    self, columns=None, include_metas=False, compute_variance=False)

    Compute basic statistics for the specified variables: minimal and maximal
    value, the mean and a varianca (or a zero placeholder), the number
    of missing and defined values.

    :param columns: a list of columns for which the statistics is computed;
        if `None`, the function computes the data for all variables
    :type columns: list of ints, variable names or descriptors of type
        :obj:`Orange.data.Variable`
    :param include_metas: a flag which tells whether to include meta attributes
        (applicable only if `columns` is `None`)
    :type include_metas: bool
    :param compute_variance: a flag which tells whether to compute the variance
    :type compute_variance: bool
    :return: a list with tuple (min, max, mean, variance, #nans, #non-nans)
        for each variable
    :rtype: list

.. method:: _compute_distributions(self, columns=None)

    Compute the distribution for the specified variables. The result is a list
    of pairs containing the distribution and the number of rows for which the
    variable value was missing.

    For discrete variables, the distribution is represented as a vector with
    absolute frequency of each value. For continuous variables, the result is
    a 2-d array of shape (2, number-of-distinct-values); the first row contains
    (distinct) values of the variables and the second has their absolute
    frequencies.

    :param columns: a list of columns for which the distributions are computed;
        if `None`, the function runs over all variables
    :type columns: list of ints, variable names or descriptors of type
        :obj:`Orange.data.Variable`
    :return: a list of distributions
    :rtype: list of numpy arrays

.. automethod:: Orange.data.storage.Storage._compute_contingency
