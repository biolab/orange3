.. currentmodule:: Orange.data.sql.table

########################
SQL table (``data.sql``)
########################

.. autoclass:: Orange.data.sql.table.SqlTable
    :members:
    :special-members:

    :obj:`SqlTable` represents a table with the data which is stored in the
    database. Besides the inherited attributes, the object stores a connection
    to the database and row filters.

    Constructor connects to the database, infers the variable types from the
    types of the columns in the database and constructs the corresponding
    domain description. Discrete and continuous variables are put among
    attributes, and string variables are meta attributes. The domain does not
    have a class.

    :obj:`SqlTable` overloads the data access methods for random access to
    rows and for iteration (`__getitem__` and `__iter__`). It also provides
    methods for fast computation of basic statistics, distributions and
    contingency matrices, as well as for filtering the data. Filtering the
    data returns a new instance of :obj:`SqlTable`. The new instances however
    differs only in that an additional filter is added to the row_filter.

    All evaluation is lazy in the sense that most operations just modify the
    domain and the list of filters. These are used to construct an SQL query
    when the data is actually needed, for instance to retrieve a data row or
    compute a distribution of values for a certain column.

    .. attribute:: connection

        The object that holds the database connection. An instance of a class
        compatible with Python DB API 2.0.

    .. attribute:: host

        The host name of the database server

    .. attribute:: database

        The name of the database

    .. attribute:: table_name

        The name of the table in the database

    .. attribute:: row_filters

        A list of filters that are applied when constructing the query. The
        filters in the should have a method `to_sql`. Module
        :obj:`Orange.data.sql.filter` contains classes derived from filters in
        :obj:`Orange.data.filter` with the appropriate implementation of the
        method.


.. autoclass:: Orange.data.sql.table.SqlRowInstance
    :members:
