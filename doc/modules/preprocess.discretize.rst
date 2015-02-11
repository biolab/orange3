.. class:: Orange.preprocess.DomainDiscretizer

    Construct a table
    Construct a domain in which discrete attributes are replaced by
    continuous. Existing continuous attributes can be normalized.

    The attributes are treated according to their types:

    * binary variables are transformed into 0.0/1.0 or -1.0/1.0
      indicator variables

    * multinomial variables are treated according to the flag
      ``multinomial_treatment``.

    * discrete attribute with only one possible value are removed;

    * continuous variables can be normalized or left unchanged.

    The typical use of the class is as follows::

        import Orange
        titanic = Orange.data.Table("titanic")
        continuizer = Orange.preprocess.DomainContinuizer()
        continuizer.multinomial_treatment = continuizer.FirstAsBase
        domain1 = continuizer(titanic)
        titanic1 = Orange.data.Table(domain1, titanic)

    Domain continuizers can be given either a data set or a domain, and return
    a new domain. When given only the domain, they cannot normalize continuous
    attributes or use the most frequent value as the base value.

    The class can also behave like a function:
    if the constructor is given the data or a domain, the constructed
    continuizer is immediately applied and the constructor returns a transformed
    domain instead of the continuizer instance::

        domain1 = Orange.preprocess.DomainContinuizer(titanic)

    By default, the class does not change continuous and class attributes,
    discrete attributes are replaced with N attributes (``Indicators``) with
    values 0 and 1.

    The class has a number of attributes that can be set either in constructor
    or, later, as attributes.

    .. attribute:: zero_based

        Determines the value used as the "low" value of the variable. When
        binary variables are transformed into continuous or when multivalued
        variable is transformed into multiple variables, the transformed
        variable can either have values 0.0 and 1.0 (default,
        ``zero_based=True``) or -1.0 and 1.0 (``zero_based=False``). This
        attribute also determines the interval for normalized continuous
        variables (either [-1, 1] or [0, 1]). (Default: ``False``)

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

               continuizer = Orange.preprocess.DomainContinuizer()
               domain1 = continuizer(titanic)
               titanic1 = Orange.data.Table(domain1, titanic)

           we have ::

               >>> titanic.domain
               [status, age, sex | survived]
               >>> domain1
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
               >>> continuizer(titanic.domain)
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
                >>> continuizer(titanic)
                [status=first, status=second, status=third, age=child, sex=female | survived]

       ``Continuize.Remove``
           Discrete variables are removed. ::

               >>> continuizer.multinomial_treatment = continuizer.Remove
               >>> continuizer(titanic)
               [ | survived]

       ``Continuize.RemoveMultinomial``
           Discrete variables with more than two values are removed. Binary
           variables are treated the same as in `FirstAsBase`.

            >>> continuizer.multinomial_treatment = continuizer.RemoveMultinomial
            >>> continuizer(titanic)
            [age=child, sex=male | survived]

       ``Continuize.ReportError``
           Raise an error if there are any multinomial variables in the data.

       ``Continuize.AsOrdinal``
           Multinomial variables are treated as ordinal and replaced by
           continuous variables with indices within
           :obj:`~Orange.data.DiscreteVariable.values`, e.g. 0, 1, 2, 3...

                >>> continuizer.multinomial_treatment = continuizer.AsOrdinal
                >>> titanic1 = data.Table(continuizer(titanic), titanic)
                >>> titanic[700]
                [third, adult, male | no]
                >>> titanic1[700]
                [3.000, 0.000, 1.000 | no]

       ``DomainContinuizer.AsNormalizedOrdinal``
           As above, except that the resulting continuous value will be from
           range 0 to 1, e.g. 0, 0.333, 0.667, 1 for a four-valued variable::

                >>> continuizer.multinomial_treatment = continuizer.AsNormalizedOrdinal
                >>> titanic1 = Orange.data.Table(continuizer(titanic), titanic)
                >>> titanic1[700]
                [1.000, 0.000, 1.000 | no]
                >>> titanic1[15]
                [0.333, 0.000, 1.000 | yes]

    .. attribute:: normalize_continuous

        If ``None``, continuous variables are left unchanged. If
        ``DomainContinuizer.NormalizeBySD``, they are replaced with
        standardized values by subtracting the average value and dividing by
        the standard deviation. Attribute ``zero_based`` has no effect on this
        standardization. If ``DomainContinuizer.NormalizeBySpan``, they are
        replaced with normalized values by subtracting min value of the data
        and dividing by span (max - min). Statistics are computed from the data,
        so constructor must be given data, not just domain. (Default: ``None``)

    .. attribute:: transform_class

        If ``True`` the class is replaced by continuous
        attributes or normalized as well. Multiclass problems are thus
        transformed to multitarget ones. (Default: ``False``)
