.. currentmodule:: Orange.data

###############################
Domain description (``domain``)
###############################

Description of a domain stores a list of features, class(es) and meta
attribute descriptors. A domain descriptor is attached to all tables in
Orange to assign names and types to the corresponding columns. Domain
descriptors are also stored in predictive models and other objects to
facilitate automated conversions between domains, as described below.

.. autoclass:: Orange.data.Domain

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

    .. automethod:: Domain.__init__

        The following script constructs a domain with a discrete feature
        *gender* and continuous feature *age*, and a continuous target *salary*.
        ::

            from Orange.data import Domain, DiscreteVariable, ContinuousVariable
            domain = Domain([DiscreteVariable("gender"), ContinuousVariable("age")],
                            ContinuousVariable("salary"))

        This constructs a new domain with some features from the Iris data set
        and a new feature *color*. ::

            from Orange.data import Domain, DiscreteVariable, Table
            iris = Table("iris")
            new_domain = Domain(["sepal length", "petal length", DiscreteVariable("color")],
                                iris.domain.class_var, source=data.domain)

    .. automethod:: from_numpy

        ::

            >>> import numpy as np
            >>> from Orange.data import Domain
            >>> X = np.arange(20, dtype=float).reshape(5, 4)
            >>> Y = np.arange(5, dtype=int)
            >>> domain = Domain.from_numpy(X, Y)
            >>> domain
            [Feature 1, Feature 2, Feature 3, Feature 4 | Class 1]

    .. automethod:: var_from_domain

        ::

            >>> from Orange.data import Table
            >>> iris = Table("iris")
            >>> domain = iris.domain
            >>> domain.var_from_domain("petal length")
            ContinuousVariable('petal length')
            >>> domain.var_from_domain(2)
            ContinuousVariable('petal length')

    .. automethod:: __getitem__

        ::

            >>> domain[1:3]
            (ContinuousVariable('sepal width'), ContinuousVariable('petal length'))

    .. automethod:: __len__
    .. automethod:: __contains__

        ::

            >>> "petal length" in domain
            True
            >>> "age" in domain
            False

    .. automethod:: index

        ::

            >>> domain.index("petal length")
            2

    .. automethod:: has_discrete_attributes
    .. automethod:: has_continuous_attributes

Domain conversion
#################

    Domain descriptors also convert data instances between different domains.

    In a typical scenario, we may want to discretize some continuous data before
    inducing a model. Discretizers (:mod:`Orange.feature.discretization`)
    construct a new data table with attribute descriptors
    (:class:`Orange.data.variable`), that include the corresponding functions
    for conversion from continuous to discrete values. The trained model stores
    this domain descriptor and uses it to convert instances from the original
    domain to the discretized one at prediction phase.

    In general, instances are converted between domains as follows.

    - If the target attribute appears in the source domain, the value is
      copied; two attributes are considered the same if they have the same
      descriptor.
    - If the target attribute descriptor defines a function for value
      transformation, the value is transformed.
    - Otherwise, the value is marked as missing.

    An exception to this rule are domains in which the anonymous flag is set.
    When the source or the target domain is anonymous, they match if they have
    the same number of variables and types. In this case, the data is copied
    without considering the attribute descriptors.

    .. automethod:: Domain.get_conversion
    .. automethod:: Domain.convert