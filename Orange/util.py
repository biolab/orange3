"""Various small utilities that might be useful everywhere"""
import os
import inspect
from enum import Enum as _Enum
from functools import wraps, partial
from operator import attrgetter
from itertools import chain, count, repeat

from collections import OrderedDict
import warnings

# Exposed here for convenience. Prefer patching to try-finally blocks
from unittest.mock import patch  # pylint: disable=unused-import

# Backwards-compat
from Orange.data.util import scale  # pylint: disable=unused-import


class OrangeWarning(UserWarning):
    pass


class OrangeDeprecationWarning(OrangeWarning, DeprecationWarning):
    pass


warnings.simplefilter('default', OrangeWarning)

if os.environ.get('ORANGE_DEPRECATIONS_ERROR'):
    warnings.simplefilter('error', OrangeDeprecationWarning)


def deprecated(obj):
    """
    Decorator. Mark called object deprecated.

    Parameters
    ----------
    obj: callable or str
        If callable, it is marked as deprecated and its calling raises
        OrangeDeprecationWarning. If str, it is the alternative to be used
        instead of the decorated function.

    Returns
    -------
    f: wrapped callable or decorator
        Returns decorator if obj was str.

    Examples
    --------
    >>> @deprecated
    ... def old():
    ...     return 'old behavior'
    >>> old()  # doctest: +SKIP
    /... OrangeDeprecationWarning: Call to deprecated ... old ...
    'old behavior'

    >>> class C:
    ...     @deprecated('C.new()')
    ...     def old(self):
    ...         return 'old behavior'
    ...     def new(self):
    ...         return 'new behavior'
    >>> C().old() # doctest: +SKIP
    /... OrangeDeprecationWarning: Call to deprecated ... C.old ...
      Instead, use C.new() ...
    'old behavior'
    """
    alternative = ('; Instead, use ' + obj) if isinstance(obj, str) else ''

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = '{}.{}'.format(
                func.__self__.__class__,
                func.__name__) if hasattr(func, '__self__') else func
            warnings.warn('Call to deprecated {}{}'.format(name, alternative),
                          OrangeDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper

    return decorator if alternative else decorator(obj)


def try_(func, default=None):
    """Try return the result of func, else return default."""
    try:
        return func()
    except Exception:
        return default


def flatten(lst):
    """Flatten iterable a single level."""
    return chain.from_iterable(lst)


class Registry(type):
    """Metaclass that registers subtypes."""
    def __new__(mcs, name, bases, attrs):
        cls = type.__new__(mcs, name, bases, attrs)
        if not hasattr(cls, 'registry'):
            cls.registry = OrderedDict()
        else:
            cls.registry[name] = cls
        return cls

    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        if cls in cls.registry.values():
            return cls.__name__
        return '{}({{{}}})'.format(cls.__name__, ', '.join(cls.registry))


def namegen(prefix='_', *args, spec_count=count, **kwargs):
    """Continually generate names with `prefix`, e.g. '_1', '_2', ..."""
    spec_count = iter(spec_count(*args, **kwargs))
    while True:
        yield prefix + str(next(spec_count))


def export_globals(globals, module_name):
    """
    Return list of important for export globals (callables, constants) from
    `globals` dict, defined in module `module_name`.

    Usage
    -----
    In some module, on the second-to-last line:

    __all__ = export_globals(globals(), __name__)

    """
    return [getattr(v, '__name__', k)
            for k, v in globals.items()                          # export
            if ((callable(v) and v.__module__ == module_name     # callables from this module
                 or k.isupper()) and                             # or CONSTANTS
                not getattr(v, '__name__', k).startswith('_'))]  # neither marked internal


_NOTSET = object()


def deepgetattr(obj, attr, default=_NOTSET):
    """Works exactly like getattr(), except that attr can be a nested attribute
    (e.g. "attr1.attr2.attr3").
    """
    try:
        return attrgetter(attr)(obj)
    except AttributeError:
        if default is _NOTSET:
            raise
        return default


def color_to_hex(color):
    return "#{:02X}{:02X}{:02X}".format(*color)


def hex_to_color(s):
    return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)


def inherit_docstrings(cls):
    """Inherit methods' docstrings from first superclass that defines them"""
    for method in cls.__dict__.values():
        if inspect.isfunction(method) and method.__doc__ is None:
            for parent in cls.__mro__[1:]:
                __doc__ = getattr(parent, method.__name__, None).__doc__
                if __doc__:
                    method.__doc__ = __doc__
                    break
    return cls


class Enum(_Enum):
    """Enum that represents itself with the qualified name, e.g. Color.red"""
    __repr__ = _Enum.__str__


def interleave(seq1, seq2):
    """
    Interleave elements of `seq2` between consecutive elements of `seq1`.

    Example
    -------
    >>> list(interleave([1, 3, 5], [2, 4]))
    [1, 2, 3, 4, 5]
    >>> list(interleave([1, 2, 3, 4], repeat("<")))
    [1, '<', 2, '<', 3, '<', 4]
    """
    iterator1, iterator2 = iter(seq1), iter(seq2)
    try:
        leading = next(iterator1)
    except StopIteration:
        pass
    else:
        for element in iterator1:
            yield leading
            try:
                yield next(iterator2)
            except StopIteration:
                return
            leading = element
        yield leading


def Reprable_repr_pretty(name, itemsiter, printer, cycle):
    # type: (str, Iterable[Tuple[str, Any]], Ipython.lib.pretty.PrettyPrinter, bool) -> None
    if cycle:
        printer.text("{0}(...)".format("name"))
    else:
        def printitem(field, value):
            printer.text(field + "=")
            printer.pretty(value)

        def printsep():
            printer.text(",")
            printer.breakable()

        itemsiter = (partial(printitem, *item) for item in itemsiter)
        sepiter = repeat(printsep)

        with printer.group(len(name) + 1, "{0}(".format(name), ")"):
            for part in interleave(itemsiter, sepiter):
                part()


class _Undef:
    def __repr__(self):
        return "<?>"
_undef = _Undef()


class Reprable:
    """A type that inherits from this class has its __repr__ string
    auto-generated so that it "[...] should look like a valid Python
    expression that could be used to recreate an object with the same
    value [...]" (see See Also section below).

    This relies on the instances of type to have attributes that
    match the arguments of the type's constructor. Only the values that
    don't match the arguments' defaults are printed, i.e.:

        >>> class C(Reprable):
        ...     def __init__(self, a, b=2):
        ...         self.a = a
        ...         self.b = b
        >>> C(1, 2)
        C(a=1)
        >>> C(1, 3)
        C(a=1, b=3)

    If Reprable instances define `_reprable_module`, that string is used
    as a fully-qualified module name and is printed. `_reprable_module`
    can also be True in which case the type's home module is used.

        >>> class C(Reprable):
        ...     _reprable_module = True
        >>> C()
        Orange.util.C()
        >>> class C(Reprable):
        ...     _reprable_module = 'something_else'
        >>> C()
        something_else.C()
        >>> class C(Reprable):
        ...     class ModuleResolver:
        ...         def __str__(self):
        ...             return 'magic'
        ...     _reprable_module = ModuleResolver()
        >>> C()
        magic.C()

    See Also
    --------
    https://docs.python.org/3/reference/datamodel.html#object.__repr__
    """
    _reprable_module = ''

    def _reprable_fields(self):
        # type: () -> Iterable[Tuple[str, Any]]
        cls = self.__class__
        sig = inspect.signature(cls.__init__)
        for param in sig.parameters.values():
            # Skip self, *args, **kwargs
            if (param.name != 'self' and
                        param.kind not in (param.VAR_POSITIONAL,
                                           param.VAR_KEYWORD)):
                yield param.name, param.default

    def _reprable_omit_param(self, name, default, value):
        if default is value:
            return True
        if type(default) is type(value):
            try:
                return default == value
            except (ValueError, TypeError):
                return False
        else:
            return False

    def _reprable_items(self):
        for name, default in self._reprable_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                value = _undef
            if not self._reprable_omit_param(name, default, value):
                yield name, default, value

    def _repr_pretty_(self, p, cycle):
        """IPython pretty print hook."""
        module = self._reprable_module
        if module is True:
            module = self.__class__.__module__

        nameparts = (([str(module)] if module else []) +
                     [self.__class__.__name__])
        name = ".".join(nameparts)
        Reprable_repr_pretty(
            name, ((f, v) for f, _, v in self._reprable_items()),
            p, cycle)

    def __repr__(self):
        module = self._reprable_module
        if module is True:
            module = self.__class__.__module__
        nameparts = (([str(module)] if module else []) +
                     [self.__class__.__name__])
        name = ".".join(nameparts)
        return "{}({})".format(
            name, ", ".join("{}={!r}".format(f, v) for f, _, v in self._reprable_items())
        )

# For best result, keep this at the bottom
__all__ = export_globals(globals(), __name__)

# ONLY NON-EXPORTED VALUES BELOW HERE
