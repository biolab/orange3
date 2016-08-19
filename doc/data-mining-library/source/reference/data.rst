.. currentmodule:: Orange.data

#####################
Data model (``data``)
#####################

Orange stores data in :obj:`Orange.data.TableBase` classes. The most commonly used
storage is :obj:`Orange.data.Table`, which stores all data in a dense :obj:`pandas.DataFrame`.

Indexing Orange tables works exactly like it does in `pandas`.
That means the fourth row of the table is fetched with `t.iloc[3]` and the
row with index 543 is fetched with `t.loc[543]`. Accessing columns works similarly:
if `t = Table('iris')`, then `t["sepal width"]` selects the `sepal width` column.
For column names without spaces, direct access works too: `t.iris` to select the target column.

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
attributes. This division applies to domain descriptions, which logically separate
a :obj:`Orange.data.Table` into three parts, corresponding to the roles.

Attributes and classes are represented with numeric values and are used in
modelling. Meta attributes contain additional data which may be of any type.
(Currently, only string values are supported in addition to continuous and
numeric.)

Indexing is inherited from :obj:`pandas`. This means using `.loc` and `.iloc`
to access rows and slice the table, the same as :obj:`pandas` does. The only
difference is that Orange uses globally unique indexing, where different instances
of :obj:`Orange.data.Table` have different :obj:`pandas` indices, used with `.loc`.
Columns are accessed either with domain variables or their names.

Orange stores data directly in the :obj:`Orange.data.Table` exactly like `pandas` does,
and the `.X`, `.Y` and `.metas` descriptors convert the raw data into a float format,
suitable for learning. Columns with :obj:`Orange.data.StringVariable` remain as strings.

.. toctree::
    :maxdepth: 2

    data.table
    data.sql
    data.domain
    data.variable

.. index:: Data
