.. py:currentmodule:: Orange.preprocess

###################################
Data Preprocessing (``preprocess``)
###################################

.. index:: preprocessing

.. index::
   single: data; preprocessing

Preprocessing module contains data processing utilities like data
discretization, continuization, imputation and transformation.

Impute
======


.. index:: discretize data
   single: feature; discretize

Discretization
==============

Discretization replaces continuous features with the corresponding categorical
features:

.. literalinclude:: code/discretization-table.py

The variable in the new data table indicate the bins to which the original
values belong. ::

    Original data set:
    [5.1, 3.5, 1.4, 0.2 | Iris-setosa]
    [4.9, 3.0, 1.4, 0.2 | Iris-setosa]
    [4.7, 3.2, 1.3, 0.2 | Iris-setosa]
    Discretized data set:
    [<5.5, >=3.2, <2.5, <0.8 | Iris-setosa]
    [<5.5, [2.8, 3.2), <2.5, <0.8 | Iris-setosa]
    [<5.5, >=3.2, <2.5, <0.8 | Iris-setosa]


Default discretization method (four bins with approximatelly equal number of
data instances) can be replaced with other methods.

.. literalinclude:: code/discretization-table-method.py
    :lines: 3-5

.. autoclass::Orange.preprocess.Discretize

..
    Transformation procedure
    ------------------------

    `Discretization Algorithms`_ return a discretized variable (with fixed
    parameters) that can transform either the learning or the testing data.
    Parameter learning is separate to the transformation, as in machine
    learning only the training set should be used to induce parameters.

    To obtain discretized features, call a discretization algorithm with
    with the data and the feature to discretize. The feature can be given
    either as an index name or :obj:`Orange.data.Variable`. The following
    example creates a discretized feature::

        import Orange
        data = Orange.data.Table("iris.tab")
        disc = Orange.feature.discretization.EqualFreq(n=4)
        disc_var = disc(data, 0)

    The values of the first attribute will be discretized the data is
    transformed to the  :obj:`Orange.data.Domain` domain that includes
    ``disc_var``.  In the example below we add the discretized first attribute
    to the original domain::

      ndomain = Orange.data.Domain([disc_var] + list(data.domain.attributes),
          data.domain.class_vars)
      ndata = Orange.data.Table(ndomain, data)
      print(ndata)

    The printout::

      [[<5.150000, 5.1, 3.5, 1.4, 0.2 | Iris-setosa],
       [<5.150000, 4.9, 3.0, 1.4, 0.2 | Iris-setosa],
       [<5.150000, 4.7, 3.2, 1.3, 0.2 | Iris-setosa],
       [<5.150000, 4.6, 3.1, 1.5, 0.2 | Iris-setosa],
       [<5.150000, 5.0, 3.6, 1.4, 0.2 | Iris-setosa],
       ...
      ]

_`Discretization Algorithms`
----------------------------

.. autoclass:: Orange.preprocess.discretize.EqualWidth

.. autoclass:: Orange.preprocess.discretize.EqualFreq

.. autoclass:: Orange.preprocess.discretize.EntropyMDL

To add a new discretization, derive it from ``Discretization``.

.. autoclass:: Orange.preprocess.discretize.Discretization

Continuization
==============

.. class:: Orange.preprocess.Continuize

    Given a data table, return a new table in which the discretize attributes
    are replaced with continuous or removed.

    * binary variables are transformed into 0.0/1.0 or -1.0/1.0
      indicator variables, depending upon the argument ``zero_based``.

    * multinomial variables are treated according to the argument
      ``multinomial_treatment``.

    * discrete attribute with only one possible value are removed;

    ::

        import Orange
        titanic = Orange.data.Table("titanic")
        continuizer = Orange.preprocess.Continuize()
        titanic1 = continuizer(titanic)

    The class has a number of attributes that can be set either in constructor
    or, later, as attributes.

    .. attribute:: zero_based

        Determines the value used as the "low" value of the variable. When
        binary variables are transformed into continuous or when multivalued
        variable is transformed into multiple variables, the transformed
        variable can either have values 0.0 and 1.0 (default,
        ``zero_based=True``) or -1.0 and 1.0 (``zero_based=False``).

    .. attribute:: multinomial_treatment

       Defines the treatment of multinomial variables.

       ``Continuize.Indicators``

           The variable is replaced by indicator variables, each
           corresponding to one value of the original variable.
           For each value of the original attribute, only the
           corresponding new attribute will have a value of one and others
           will be zero. This is the default behaviour.

           Note that these variables are not independent, so they cannot be
           used (directly) in, for instance, linear or logistic regression.

           For example, data set "titanic" has feature "status" with
           values "crew", "first", "second" and "third", in that order. Its
           value for the 15th row is "first". Continuization replaces the
           variable with variables "status=crew", "status=first",
           "status=second" and "status=third". After ::

               continuizer = Orange.preprocess.Continuize()
               titanic1 = continuizer(titanic)

           we have ::

               >>> titanic.domain
               [status, age, sex | survived]
               >>> titanic1.domain
               [status=crew, status=first, status=second, status=third,
                age=adult, age=child, sex=female, sex=male | survived]

           For the 15th row, the variable "status=first" has value 1 and the
           values of the other three variables are 0::

               >>> print(titanic[15])
               [first, adult, male | yes]
               >>> print(titanic1[15])
               [0.000, 1.000, 0.000, 0.000, 1.000, 0.000, 0.000, 1.000 | yes]


       ``Continuize.FirstAsBase``
           Similar to the above, except that it creates indicators for all
           values except the first one, according to the order in the variable's
           :obj:`~Orange.data.DiscreteVariable.values` attribute. If all
           indicators in the transformed data instance are 0, the original
           instance had the first value of the corresponding variable.

           If the variable descriptor defines the
           :obj:`~Orange.data.DiscreteVariable.base_value`, the
           specified value is used as base instead of the first one.

           Continuizing the variable "status" with this setting gives variables
           "status=first", "status=second" and "status=third". If all of them
           were 0, the status of the original data instance was "crew".

               >>> continuizer.multinomial_treatment = continuizer.FirstAsBase
               >>> continuizer(titanic).domain
               [status=first, status=second, status=third, age=child, sex=male | survived]

       ``Continuize.FrequentAsBase``
           Like above, except that the most frequent value is used as the
           base. If there are multiple most frequent values, the
           one with the lowest index in
           :obj:`~Orange.data.DiscreteVariable.values` is used. The frequency
           of values is extracted from data, so this option does not work if
           only the domain is given.

           Continuizing the Titanic data in this way differs from the above by
           the attributes sex: instead of "sex=male" it constructs "sex=female"
           since there were more females than males on Titanic. ::

                >>> continuizer.multinomial_treatment = continuizer.FrequentAsBase
                >>> continuizer(titanic).domain
                [status=first, status=second, status=third, age=child, sex=female | survived]

       ``Continuize.Remove``
           Discrete variables are removed. ::

               >>> continuizer.multinomial_treatment = continuizer.Remove
               >>> continuizer(titanic).domain
               [ | survived]

       ``Continuize.RemoveMultinomial``
           Discrete variables with more than two values are removed. Binary
           variables are treated the same as in `FirstAsBase`.

            >>> continuizer.multinomial_treatment = continuizer.RemoveMultinomial
            >>> continuizer(titanic).domain
            [age=child, sex=male | survived]

       ``Continuize.ReportError``
           Raise an error if there are any multinomial variables in the data.

       ``Continuize.AsOrdinal``
           Multinomial variables are treated as ordinal and replaced by
           continuous variables with indices within
           :obj:`~Orange.data.DiscreteVariable.values`, e.g. 0, 1, 2, 3...

                >>> continuizer.multinomial_treatment = continuizer.AsOrdinal
                >>> titanic1 = continuizer(titanic)
                >>> titanic[700]
                [third, adult, male | no]
                >>> titanic1[700]
                [3.000, 0.000, 1.000 | no]

       ``Continuize.AsNormalizedOrdinal``
           As above, except that the resulting continuous value will be from
           range 0 to 1, e.g. 0, 0.333, 0.667, 1 for a four-valued variable::

                >>> continuizer.multinomial_treatment = continuizer.AsNormalizedOrdinal
                >>> titanic1 = continuizer(titanic)
                >>> titanic1[700]
                [1.000, 0.000, 1.000 | no]
                >>> titanic1[15]
                [0.333, 0.000, 1.000 | yes]

    .. attribute:: transform_class

        If ``True`` the class is replaced by continuous
        attributes or normalized as well. Multiclass problems are thus
        transformed to multitarget ones. (Default: ``False``)



.. class:: Orange.preprocess.DomainContinuizer

    Construct a domain in which discrete attributes are replaced by
    continuous. ::

        domain_continuizer = Orange.preprocess.DomainContinuizer()
        domain1 = domain_continuizer(titanic)

    :obj:`Orange.preprocess.Continuize` calls `DomainContinuizer` to construct
    the domain.

    Domain continuizers can be given either a data set or a domain, and return
    a new domain. When given only the domain, use the most frequent value as
    the base value.

    The class can also behave like a function: if the constructor is given the
    data or a domain, the constructed continuizer is immediately applied and
    the constructor returns a transformed domain instead of the continuizer
    instance::

        domain1 = Orange.preprocess.DomainContinuizer(titanic)

    By default, the class does not change continuous and class attributes,
    discrete attributes are replaced with N attributes (``Indicators``) with
    values 0 and 1.

Normalization
=============

.. autoclass:: Orange.preprocess.Normalize


Randomization
=============

.. autoclass:: Orange.preprocess.Randomize


Remove
======

.. autoclass:: Orange.preprocess.Remove

Feature selection
=================

`Feature scoring`
-----------------

Feature score is an assessment of the usefulness of the feature for
prediction of the dependant (class) variable. Orange provides classes
that compute the common feature scores for classification and regression.

The code below computes the information gain of feature "tear_rate"
in the Lenses data set:

    >>> data = Orange.data.Table("lenses")
    >>> Orange.preprocess.score.InfoGain("tear_rate", data)
    0.54879494069539858

An alternative way of invoking the scorers is to construct the scoring
object, like in the following example:

    >>> gain = Orange.preprocess.score.InfoGain()
    >>> for att in data.domain.attributes:
    ...     print("%s %.3f" % (att.name, gain(att, data)))
    age 0.039
    prescription 0.040
    astigmatic 0.377
    tear_rate 0.549

Feature scoring methods work on different feature types (continuous or discrete)
and different types of target variables (classification or regression problem).
Refer to method's `feature_type` and `class_type` attributes for intended type
or employ preprocessing methods (e.g. discretization) for conversion between
data types.

.. autoclass:: Orange.preprocess.score.ANOVA

.. autoclass:: Orange.preprocess.score.Chi2

.. autoclass:: Orange.preprocess.score.GainRatio

.. autoclass:: Orange.preprocess.score.Gini

.. autoclass:: Orange.preprocess.score.InfoGain

.. autoclass:: Orange.preprocess.score.UnivariateLinearRegression

`Feature selection`
-------------------

We can use feature selection to limit the analysis to only the most relevant
or informative features in the data set.

Feature selection with a scoring method that works on continuous features will
retain all discrete features and vice versa.

The code below constructs a new data set consisting of two best features
according to the ANOVA method:

    >>> data = Orange.data.Table("wine")
    >>> anova = Orange.preprocess.score.ANOVA()
    >>> selector = Orange.preprocess.SelectBestFeatures(method=anova, k=2)
    >>> data2 = selector(data)
    >>> data2.domain
    [Flavanoids, Proline | Wine]

.. autoclass:: Orange.preprocess.SelectBestFeatures

Preprocessors
=============
