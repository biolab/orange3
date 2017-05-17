.. currentmodule:: Orange.data

#####################
Data model (``data``)
#####################

Orange stores data in :obj:`Orange.data.Storage` classes. The most commonly used
storage is :obj:`Orange.data.Table`, which stores all data in two-dimensional
numpy arrays. Each row of the data represents a data instance.

Individual data instances are represented as instances of
:obj:`Orange.data.Instance`. Different storage classes may derive subclasses
of :obj:`~Orange.data.Instance` to represent the retrieved rows in the data
more efficiently and to allow modifying the data through modifying data
instance. For example, if `table` is :obj:`Orange.data.Table`, `table[0]`
returns the row as :obj:`Orange.data.RowInstance`.

Every storage class and data instance has an associated domain description
`domain` (an instance of :obj:`Orange.data.Domain`) that stores descriptions of
data columns. Every column is described by an instance of a class derived from
:obj:`Orange.data.Variable`. The subclasses correspond to continuous variables
(:obj:`Orange.data.ContinuousVariable`), discrete variables
(:obj:`Orange.data.DiscreteVariable`) and string variables
(:obj:`Orange.data.StringVariable`). These descriptors contain the
variable's name, symbolic values, number of decimals in printouts and similar.

The data is divided into attributes (features, independent variables), class
variables (classes, targets, outcomes, dependent variables) and meta
attributes. This division applies to domain descriptions, data storages that
contain separate arrays for each of the three parts of the data and data
instances.

Attributes and classes are represented with numeric values and are used in
modelling. Meta attributes contain additional data which may be of any type.
(Currently, only string values are supported in addition to continuous and
numeric.)

In indexing, columns can be referred to by their names,
descriptors or an integer index. For example, if `inst` is a data instance
and `var` is a descriptor of type :obj:`~Orange.data.Continuous`, referring to
the first column in the data, which is also names "petal length", then
`inst[var]`, `inst[0]` and `inst["petal length"]` refer to the first value
of the instance. Negative indices are used for meta attributes, starting with
-1.

Continuous and discrete values can be represented by any numerical type; by
default, Orange uses double precision (64-bit) floats. Discrete values are
represented by whole numbers.

.. toctree::
    :maxdepth: 2

    data.storage
    data.table
    data.sql
    data.domain
    data.variable
    data.value
    data.instance
    data.filters
    data.io

.. index:: Data
