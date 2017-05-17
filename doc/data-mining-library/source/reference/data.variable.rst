.. currentmodule:: Orange.data

###################################
Variable Descriptors (``variable``)
###################################

Every variable is associated with a descriptor that stores its name
and other properties. Descriptors serve three main purposes:

- conversion of values from textual format (e.g. when reading files) to
  the internal representation and back (e.g. when writing files or printing
  out);

- identification of variables: two variables from different data sets are
  considered to be the same if they have the same descriptor;

- conversion of values between domains or data sets, for instance from
  continuous to discrete data, using a pre-computed transformation.

Descriptors are most often constructed when loading the data from files. ::

    >>> from Orange.data import Table
    >>> iris = Table("iris")

    >>> iris.domain.class_var
    DiscreteVariable('iris')
    >>> iris.domain.class_var.values
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    >>> iris.domain[0]
    ContinuousVariable('sepal length')
    >>> iris.domain[0].number_of_decimals
    1

Some variables are derived from others. For instance, discretizing a continuous
variable gives a new, discrete variable. The new variable can compute its
values from the original one.

    >>> from Orange.preprocess import DomainDiscretizer
    >>> discretizer = DomainDiscretizer()
    >>> d_iris = discretizer(iris)
    >>> d_iris[0]
    DiscreteVariable('D_sepal length')
    >>> d_iris[0].values
    ['<5.2', '[5.2, 5.8)', '[5.8, 6.5)', '>=6.5']

See :obj:`Variable.compute_value` for a detailed explanation.

Constructors
------------

Orange maintains lists of existing descriptors for variables. This facilitates
the reuse of descriptors: if two data sets refer to the same variables, they
should be assigned the same descriptors so that, for instance, a model
trained on one data set can make predictions for the other.

Variable descriptors are seldom constructed in user scripts. When needed,
this can be done by calling the constructor directly or by calling the class
method `make`. The difference is that the latter returns an existing
descriptor if there is one with the same name and which matches the other
conditions, such as having the prescribed list of discrete values for
:obj:`~Orange.data.DiscreteVariable`::

    >>> from Orange.data import ContinuousVariable
    >>> age = ContinuousVariable.make("age")
    >>> age1 = ContinuousVariable.make("age")
    >>> age2 = ContinuousVariable("age")
    >>> age is age1
    True
    >>> age is age2
    False

The first line returns a new descriptor after not finding an existing desciptor
for a continuous variable named "age". The second reuses the first descriptor.
The last creates a new one since the constructor is invoked directly.

The distinction does not matter in most cases, but it is important when
loading the data from different files. Orange uses the `make` constructor
when loading data.

Base class
----------

.. autoclass:: Variable

    .. automethod:: is_primitive
    .. automethod:: str_val
    .. automethod:: to_val
    .. automethod:: val_from_str_add
    .. autoattribute:: compute_value

    Method `compute_value` is usually invoked behind the scenes in
    conversion of domains::

        >>> from Orange.data import Table
        >>> from Orange.preprocess import DomainDiscretizer

        >>> iris = Table("iris")
        >>> iris_1 = iris[::2]
        >>> discretizer = DomainDiscretizer()
        >>> d_iris_1 = discretizer(iris_1)

        >>> d_iris_1[0]
        DiscreteVariable('D_sepal length')
        >>> d_iris_1[0].source_variable
        ContinuousVariable('sepal length')
        >>> d_iris_1[0].compute_value
        <Orange.feature.discretization.Discretizer at 0x10d5108d0>

    The data is loaded and the instances on even places are put into a new
    table, from which we compute discretized data. The discretized variable
    "D_sepal length" refers to the original as its source and stores a function
    for conversion of the original continuous values into the discrete.
    This function (and the corresponding functions for other variables)
    is used for converting the remaining data::

        >>> iris_2 = iris[1::2]
        >>> d_iris_2 = Table(d_iris_1.domain, iris_2)
        >>> d_iris_2[0]
        [<5.2, [2.8, 3), <1.6, <0.2 | Iris-setosa]

    In the first line we select the instances with odd indices in the original
    table, that is, the data which was not used for computing the
    discretization. In the second line we construct a new data table with the
    discrete domain `d_iris_1.domain` and using the original data `iris_2`.
    Behind the scenes, the values for those variables in the destination domain
    (`d_iris_1.domain`) that do not appear in the source domain
    (`iris_2.domain`) are computed by passing the source data instance to the
    destination variables' :obj:`Variable.compute_value`.

    This mechanism is used throughout Orange to compute all preprocessing on
    training data and applying the same transformations on the testing data
    without hassle.

    Note that even such conversions are typically not coded in user scripts
    but implemented within the provided wrappers and cross-validation
    schemes.

Continuous variables
--------------------

.. autoclass:: ContinuousVariable


    .. automethod:: make
    .. automethod:: is_primitive
    .. automethod:: str_val
    .. automethod:: to_val
    .. automethod:: val_from_str_add

Discrete variables
------------------

.. autoclass:: DiscreteVariable

    .. automethod:: make
    .. automethod:: is_primitive
    .. automethod:: str_val
    .. automethod:: to_val
    .. automethod:: val_from_str_add

String variables
----------------

.. autoclass:: StringVariable

    .. automethod:: make
    .. automethod:: is_primitive
    .. automethod:: str_val
    .. automethod:: to_val
    .. automethod:: val_from_str_add

Time variables
--------------
Time variables are continuous variables with value 0 on the Unix epoch,
1 January 1970 00:00:00.0 UTC. Positive numbers are dates beyond this date,
and negative dates before. Due to limitation of Python :py:mod:`datetime` module,
only dates in 1 A.D. or later are supported.

.. autoclass:: TimeVariable

    .. automethod:: parse
