"""Various small utilities that might be useful everywhere"""

from functools import wraps
from itertools import chain, count
from collections import OrderedDict
import warnings

import numpy as np


class OrangeWarning(UserWarning):
    pass


class OrangeDeprecationWarning(OrangeWarning, DeprecationWarning):
    pass


warnings.simplefilter('default', OrangeWarning)


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
            name = '{}.{}'.format(func.__self__.__class__, func.__name__) if hasattr(func, '__self__') else func
            warnings.warn('Call to deprecated {}{}'.format(name, alternative),
                          OrangeDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper

    return decorator if alternative else decorator(obj)


def flatten(lst):
    """Flatten iterable a single level."""
    return chain.from_iterable(lst)


def scale(values, min=0, max=1):
    """Return values scaled to [min, max]"""
    ptp = np.nanmax(values) - np.nanmin(values)
    if ptp == 0:
        return np.clip(values, min, max)
    return (-np.nanmin(values) + values) / ptp * (max - min) + min


def abstract(obj):
    """Designate decorated class or method abstract."""
    if isinstance(obj, type):
        warnings.warn('Marking types @abstract not supported ({})'.format(obj.__name__),
                      OrangeWarning, stacklevel=2)
        return obj
    else:
        if not hasattr(obj, '__qualname__'):
            raise TypeError('Put @abstract decorator below (evaluated before) '
                            'any of @staticmethod, @classmethod, or @property.')
        cls_name = obj.__qualname__.rsplit('.', 1)[0]

        @wraps(obj)
        def _refuse__call__(*args, **kwargs):
            raise NotImplementedError("Can't call abstract method {} of class {}"
                                      .format(obj.__name__, cls_name))

        return _refuse__call__


class Registry(type):
    """Metaclass that registers subtypes."""
    def __new__(cls, name, bases, attrs):
        obj = type.__new__(cls, name, bases, attrs)
        if not hasattr(cls, 'registry'):
            cls.registry = OrderedDict()
        else:
            cls.registry[name] = obj
        return obj

    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        if cls in cls.registry.values():
            return cls.__name__
        return '{}({{{}}})'.format(cls.__name__, ', '.join(cls.registry))


def namegen(prefix='_', *args, count=count, **kwargs):
    """Continually generate names with `prefix`, e.g. '_1', '_2', ..."""
    count = iter(count(*args, **kwargs))
    while True:
        yield prefix + str(next(count))


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


def color_to_hex(color):
    return "#{:02X}{:02X}{:02X}".format(*color)


def hex_to_color(s):
    return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)

# For best result, keep this at the bottom
__all__ = export_globals(globals(), __name__)

# ONLY NON-EXPORTED VALUES BELOW HERE
