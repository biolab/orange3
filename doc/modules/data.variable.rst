.. currentmodule:: Orange.data

###################################
Variable Descriptors (``variable``)
###################################

Every variable is associated with a descriptor that stores its name, type
and other properties, and takes care of conversion of values from textual
format to floats and back.

Derived classes store lists of existing descriptors for variables of each
particular type. The provide a supplementary constructor :obj:`make`, which
returns an existing variable instead of creating a new one, when possible.

Descriptors facilitate computation of variables from other variables. This
is used in domain conversion: if the destination domain contains a variable
that is not present in the original domain, the variables value is computed
by calling its method :obj:`compute_value`, passing the data instance from
the original domain as an argument. Method :obj:`compute_value` calls
:obj:`get_value_from` if defined and returns its result. If the variables
does not have :obj:`get_value_from`, :obj:`compute_value` returns
:obj:`Unknown`.

.. autoclass:: Variable
    :members:

.. autoclass:: DiscreteVariable
    :members:

.. autoclass:: ContinuousVariable
    :members:

.. autoclass:: StringVariable
    :members: