.. currentmodule:: Orange.data.continuizer

##################################################
Continuization and normalization (``continuizer``)
##################################################

.. class:: Orange.data.continuizer.DomainContinuizer

    Construct a domain in which discrete attributes are replaced by
    continuous; existing continuous attributes can be normalized.

    The attributes are treated according to their types:

    * binary variables are transformed into 0.0/1.0 or -1.0/1.0
      indicator variables

    * multinomial variables are treated according to the flag
      ``multinomial_treatment``.

    * discrete attribute with less than two possible values are removed;

    * continuous variables can be normalized or left unchanged

    The typical use of the class is as follows::

        continuizer = Orange.data.continuization.DomainContinuizer()
        continuizer.multinomial_treatment = continuizer.LowestIsBase
        domain0 = continuizer(data)
        data0 = Orange.data.Table(domain0, data)

    Domain continuizers can be given either a data set or a domain, and return
    a new domain. When given only the domain, they cannot normalize continuous
    attributes or used the most frequent value as the base value, and will
    raise an exception with such settings.

    Constructor can also be passed data or domain, in which case the constructed
    continuizer is immediately applied to the data or domain and the transformed
    domain is returned instead of the continuizer instance.

    By default, the class does not change continuous and class attributes,
    discrete attributes are replaced with N attributes (``NValues``) with values
    0 and 1.

    .. attribute:: zero_based

        Determines the value used as the "low" value of the variable. When
        binary variables are transformed into continuous, when multivalued
        variable is transformed into multiple variables, the transformed
        variable can either have values 0.0 and 1.0 (default, ``zero_based``
        is ``True``) or -1.0 and 1.0 (``zero_based`` is ``False``). This
        attribute also determines the interval for normalized continuous
        variables (either [-1, 1] or [0, 1]). The
        following text assumes the default case. (Default: ``False``)

    .. attribute:: multinomial_treatment

       Decides the treatment of multinomial variables. Let N be the
       number of the variables's values. (Default: ``NValues``)

       ``DomainContinuizer.NValues``

           The variable is replaced by N indicator variables, each
           corresponding to one value of the original variable.
           For each value of the original attribute, only the
           corresponding new attribute will have a value of one and others
           will be zero.

           Note that these variables are not independent, so they cannot be
           used (directly) in, for instance, linear or logistic regression.

           For example, data set "bridges" has feature "RIVER" with
           values "M", "A", "O" and "Y", in that order. Its value for
           the 15th row is "M". Continuization replaces the variable
           with variables "RIVER=M", "RIVER=A", "RIVER=O" and
           "RIVER=Y". For the 15th row, the first has value 1 and
           others are 0.

       ``DomainContinuizer.LowestIsBase``
           Similar to the above except that it creates only N-1
           variables. The missing indicator belongs to the lowest value:
           when the original variable has the lowest value all indicators
           are 0.

           If the variable descriptor has the ``base_value`` defined, the
           specified value is used as base instead of the lowest one.

           Continuizing the variable "RIVER" gives similar results as
           above except that it would omit "RIVER=M"; all three
           variables would be zero for the 15th data instance.

       ``DomainContinuizer.FrequentIsBase``
           Like above, except that the most frequent value is used as the
           base. If there are multiple most frequent values, the
           one with the lowest index is used. The frequency of values is
           extracted from data, so this option cannot be used if constructor
           is given only a domain.

           Variable "RIVER" would be continuized similarly to above
           except that it omits "RIVER=A", which is the most frequent value.

       ``DomainContinuizer.Ignore``
           Discrete variables are omitted.

       ``DomainContinuizer.IgnoreMulti``
           Discrete variables with more than two values are omitted; two-valued
           are treated the same as in LowestIsBase.

       ``DomainContinuizer.ReportError``
           Raise an error if there are any multinominal variables in the data.

       ``DomainContinuizer.AsOrdinal``
           Multivalued variables are treated as ordinal and replaced by a
           continuous variables with the values' index, e.g. 0, 1, 2, 3...

       ``DomainContinuizer.AsNormalizedOrdinal``
           As above, except that the resulting continuous value will be from
           range 0 to 1, e.g. 0, 0.25, 0.5, 0.75, 1 for a five-valued
           variable.

    .. attribute:: normalize_continuous

        If ``None``, continuous variables are left unchanged. If
        ``DomainContinuizer.NormalizeBySD``, they are replaced with standardized values by subtracting
        the average value and dividing by the standard deviation. Attribute ``zero_based`` has no effect on this
        standardization. If ``DomainContinuizer.NormalizeBySpan``, they are replaced with normalized values by
        subtracting min value of the data and dividing by span (max - min). Statistics are computed from the data,
        so constructor must be given data, not just domain. (Default: ``None``)

    .. attribute:: transform_class

        If ``True`` the class is replaced by continuous
        attributes or normalized as well. Multiclass problems are thus
        transformed to multitarget ones. (Default: ``False``)