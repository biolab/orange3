import inspect


class WrapperMeta(type):
    """
    Meta class for scikit-learn wrapper classes.

    This is used for docstring generation/templating upon
    class definition.
    The client (class using this meta class) should define a class
    attribute `__wraps__` which contains the wrapped class.

    For instance::

        >>> class Bar:
        ...    '''I am a Bar'''

        >>> class Foo(metaclass=WrapperMeta):
        ...     __wraps__ = Bar

        >>> Foo.__doc__
        '...A wrapper for ...Bar`. The following is its
            documentation:...I am a Bar...'

    The client can also define a template for the docstring

        >>> class Foo(metaclass=WrapperMeta):
        ...     '''
        ...     Here is what ${sklname} says about itself:
        ...     ${skldoc}
        ...     '''
        ...     __wraps__ = Bar

        >>> Bar.__doc__
        'I am a Bar'

        >>> Foo.__doc__
        '...Here is what ...Bar says about itself:...I am a Bar...'

    """
    def __new__(cls, name, bases, dict_):
        cls = type.__new__(cls, name, bases, dict_)
        wrapped = getattr(cls, "__wraps__", getattr(cls, "__wrapped__", None))
        if wrapped is not None:
            doc = cls.__doc__ or """
A wrapper for `${sklname}`. The following is its documentation:

${skldoc}
            """
            sklname = "{}.{}".format(inspect.getmodule(wrapped).__name__,
                                     wrapped.__name__)
            skldoc = inspect.getdoc(wrapped) or ''
            # FIXME: make sure skl-extended classes are API-compatible
            if "Attributes\n---------" in skldoc:
                skldoc = skldoc[:skldoc.index('Attributes\n---------')]
            if "Examples\n--------" in skldoc:
                skldoc = skldoc[:skldoc.index('Examples\n--------')]
            if "Parameters\n---------" in skldoc:
                skldoc = skldoc[:skldoc.index('Parameters\n---------')]
            cls.__doc__ = (doc
                           .replace('${sklname}', sklname)
                           .replace('${skldoc}', inspect.cleandoc(skldoc)))
        return cls
