import inspect
import string


class WrapperMeta(type):
    """
    Meta class for scikit-learn wrapper classes.

    This is used for docstring generation/templating upon
    class definition.
    The client (class using this meta class) should define a class
    attribute `__wraps__` which contains the wrapped class.

    For instance::

        >>> class Foo(metaclass=WrappedMeta):
        ...     __wrapped__ = Bar
        ...

        >>> print(Foo.__doc__)
        A wrapper for `Bar`

        .. seealso: Bar

    The client can also define a template for the docstring

        >>> class Foo(metaclass=WrappedMeta):
        ...    '''
        ...    Here is what ${sklname} says about itself
        ...    ${skldoc}
        ...    '''
        ...     __wrapped__ = Bar
        ...

        >>> print(Bar.__doc__)
        I am a Bar

        >>> print(Foo.__doc__)
        Here is what Bar says about itself
        I am a Bar

    """
    class DocTemplate(string.Template):
        pattern = r"""
            \$(?:
              (?P<escaped>\$)         |  # escape (double $$)
              (?P<named>\A(?!x)x)     |  # never match unbraced identifiers
              {(?P<braced>[_a-z]\w*)} |  # braced identifier
              (?P<invalid>)           |  # invalid anything else
            )
        """

    def __new__(cls, name, bases, dict_):
        docstring = dict_.pop("__doc__", None)
        cls = type.__new__(cls, name, bases, dict_)

        skl_wrapped = getattr(cls, "__wraps__", None)

        if docstring is None and skl_wrapped is not None:
            docstring = """
            A wrapper for `${sklname}`. The following is the documentation
            from `scikit-learn <http://scikit-learn.org>`_.

            ${skldoc}

            Additional Orange parameters:

            preprocessors : list, optional (default="[]")
                An ordered list of preprocessors applied to data before
                training or testing.
        """

        if docstring is not None and skl_wrapped is not None:
            docstring = WrapperMeta.format_docstring(docstring, skl_wrapped)
            cls.__doc__ = docstring

        return cls

    @staticmethod
    def format_docstring(doc, sklclass):
        module = inspect.getmodule(sklclass)
        # TODO: prettify the name (pull the class up if it is imported at
        # a higher level and included in __all__, like ipython's help)
        sklname = "{}.{}".format(module.__name__, sklclass.__name__)
        skldoc = inspect.getdoc(sklclass)
        if "Attributes\n---------" in skldoc:
            skldoc = skldoc[:skldoc.index('Attributes\n---------')]

        mapping = {"sklname": sklname}
        if skldoc is not None:
            mapping["skldoc"] = skldoc

        doc = inspect.cleandoc(doc)
        template = WrapperMeta.DocTemplate(doc)
        return template.safe_substitute(mapping)