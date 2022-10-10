.. currentmodule:: Orange.data

###################################
Variable Descriptors (``variable``)
###################################

Every variable is associated with a descriptor that stores its name
and other properties. Descriptors serve three main purposes:

- conversion of values from textual format (e.g. when reading files) to
  the internal representation and back (e.g. when writing files or printing
  out);

- identification of variables: two variables from different datasets are
  considered to be the same if they have the same descriptor;

- conversion of values between domains or datasets, for instance from
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

See `Derived variables`_ for a detailed explanation.

Constructors
------------

Orange maintains lists of existing descriptors for variables. This facilitates
the reuse of descriptors: if two datasets refer to the same variables, they
should be assigned the same descriptors so that, for instance, a model
trained on one dataset can make predictions for the other.

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
only dates in 1 A.D. or later are supported. Note that Orange's Table stores datetime 
values as UNIX epoch (seconds from 1970-01-01), thus :obj:`Table.from_numpy` expects values in this format.

Orange's `TimeVariable` supports storing either date, time, or a combination of both:

- `TimeVariable("Timestamp", have_date=True)` stores only date information -- it is analogous to `datetime.date`

- `TimeVariable("Timestamp", have_time=True)` stores only time information (without date) -- it is analogous to `datetime.time``

- `TimeVariable("Timestamp", have_time=True, have_date=True)` stores date and time -- it is analogous to `datetime.datetime`

When the `parse` method is used to parse datetimes from a string, it is not necessary
to set the `have_time` and `have_date` attributes since they will be inferred from
from datetimes.

.. autoclass:: TimeVariable

    .. automethod:: parse

Derived variables
-----------------

The :obj:`~Variable.compute_value` mechanism is used throughout Orange to
compute all preprocessing on training data and applying the same
transformations to the testing data without hassle.

Method `compute_value` is usually invoked behind the scenes in
conversion of domains. Such conversions are are typically implemented
within the provided wrappers and cross-validation schemes.


Derived variables in Orange
```````````````````````````

Orange saves variable transformations into the domain as
:obj:`~Variable.compute_value` functions. If Orange was not using
:obj:`~Variable.compute_value`, we would have to manually transform
the data::

    >>> from Orange.data import Domain, ContinuousVariable
    >>> data = Orange.data.Table("iris")
    >>> train = data[::2]  # every second row
    >>> test = data[1::2]  # every other second instance

We will create a new data set with a single feature, "petals", that will be a
sum of petal lengths and widths::

    >>> petals = ContinuousVariable("petals")
    >>> derived_train = train.transform(Domain([petals],
    ...                                 data.domain.class_vars))
    >>> derived_train.X = train[:, "petal width"].X + \
    ...                   train[:, "petal length"].X

We have set :obj:`~Orange.data.Table`'s `X` directly. Next, we build and evaluate
a classification tree::

    >>> learner = Orange.classification.TreeLearner()
    >>> from Orange.evaluation import CrossValidation, TestOnTestData
    >>> res = CrossValidation(derived_train, [learner], k=5)
    >>> Orange.evaluation.scoring.CA(res)[0]
    0.88
    >>> res = TestOnTestData(derived_train, test, [learner])
    >>> Orange.evaluation.scoring.CA(res)[0]
    0.3333333333333333

A classification tree shows good accuracy with cross validation, but not on
separate test data, because Orange can not reconstruct the "petals"
feature for test data---we would have to reconstruct it ourselves.
But if we define :obj:`~Variable.compute_value` and therefore store the
transformation in the domain, Orange could transform both training and test data::

    >>> petals = ContinuousVariable("petals",
    ...    compute_value=lambda data: data[:, "petal width"].X + \
    ...                               data[:, "petal length"].X)
    >>> derived_train = train.transform(Domain([petals],
                                        data.domain.class_vars))
    >>> res = TestOnTestData(derived_train, test, [learner])
    >>> Orange.evaluation.scoring.CA(res)[0]
    0.9733333333333334

All preprocessors in Orange use :obj:`~Variable.compute_value`.

Example with discretization
```````````````````````````

The following example converts features to discrete::

    >>> iris = Orange.data.Table("iris")
    >>> iris_1 = iris[::2]
    >>> discretizer = Orange.preprocess.DomainDiscretizer()
    >>> d_iris_1 = discretizer(iris_1)

A dataset is loaded and a new table with every second instance is created.
On this dataset, we compute discretized data, which uses the same data
to set proper discretization intervals.

The discretized variable "D_sepal length" stores a function that can derive
continous values into discrete::

    >>> d_iris_1[0]
    DiscreteVariable('D_sepal length')
    >>> d_iris_1[0].compute_value
    <Orange.feature.discretization.Discretizer at 0x10d5108d0>

The function is used for converting the remaining data (as automatically
happens within model validation in Orange)::

    >>> iris_2 = iris[1::2]  # previously unselected
    >>> d_iris_2 = iris_2.transform(d_iris_1.domain)
    >>> d_iris_2[0]
    [<5.2, [2.8, 3), <1.6, <0.2 | Iris-setosa]

The code transforms previously unused data into the discrete domain
`d_iris_1.domain`. Behind the scenes, the values for the destination
domain that are not yet in the source domain (`iris_2.domain`) are computed
with the destination variables' :obj:`~Variable.compute_value`.

Optimization for repeated computation
`````````````````````````````````````

Some transformations share parts of computation across variables. For
example, :obj:`~Orange.projection.pca.PCA` uses all input features to
compute the PCA transform. If each output PCA component was implemented with
ordinary :obj:`~Variable.compute_value`, the PCA transform would be repeatedly
computed for each PCA component. To avoid repeated computation,
set :obj:`~Variable.compute_value` to a subclass of
:obj:`~Orange.data.util.SharedComputeValue`.

.. autoclass:: Orange.data.util.SharedComputeValue

    .. automethod:: compute

The following example creates normalized features that divide values
by row sums and then tranforms the data. In the example the function
`row_sum` is called only once; if we did not use
:obj:`~Orange.data.util.SharedComputeValue`, `row_sum` would be called
four times, once for each feature.

::

    iris = Orange.data.Table("iris")

    def row_sum(data):
        return data.X.sum(axis=1, keepdims=True)

    class DivideWithMean(Orange.data.util.SharedComputeValue):

        def __init__(self, var, fn):
            super().__init__(fn)
            self.var = var

        def compute(self, data, shared_data):
            return data[:, self.var].X / shared_data

    divided_attributes = [
        Orange.data.ContinuousVariable(
            "Divided " + attr.name,
            compute_value=DivideWithMean(attr, row_sum)
        ) for attr in iris.domain.attributes]

    divided_domain = Orange.data.Domain(
        divided_attributes,
        iris.domain.class_vars
    )

    divided_iris = iris.transform(divided_domain)

