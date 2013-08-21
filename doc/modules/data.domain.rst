.. currentmodule:: Orange.data

###############################
Domain description (``domain``)
###############################

.. autoclass:: Orange.data.Domain
    :members:

    Description of a domain. It stores a list of attribute, class and meta
    attribute descriptors with methods for indexing, printing out and
    conversions between domain. All lists are constant.

    .. attribute:: attributes

        A list of descriptors (instances of :class:`Orange.data.Variable`)
        for attributes (features, independent variables).

    .. attribute:: class_vars

        A list of descriptors for class attributes (outcomes, dependent
        variables).

    .. attribute:: variables

        A list of attributes and class attributes (the concatenation of
        the above).

    .. attribute:: class_var

        Class variable if the domain has a single class; `None` otherwise.

    .. attribute:: metas

        List of meta attributes.

    .. attribute:: indices

        A dictionary that maps variable names into indices. It includes
        ordinary attributes, class variables and meta attributes

    .. attribute:: anonymous

        `True` if the domain was constructed when converting numpy array to
        :class:`Orange.data.Table`. Such domains can be converted to and
        from other domains even if they consist of different variable
        descriptors for as long as their number and types match.

    .. attribute:: known_domains

        A weak dictionary containing instances of :class:`DomainConversion` for
        conversion from other domains into this one. Source domains are used
        as keys. The dictionary is used for caching the conversions constructed
        by :obj:`get_conversion`.

    .. attribute:: last_conversion

        The last conversion returned by :obj:`get_conversion`. Most
        conversions into the domain use the same domain as the source, so
        storing the last conversions saves a lookup into
        :obj:`known_domains`.
