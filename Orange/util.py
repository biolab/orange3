"""Various small utilities that might be useful everywhere"""

from functools import wraps
from itertools import chain
from collections import OrderedDict
import logging

import numpy as np


log = logging.getLogger()


def deprecated(obj):
    """Mark called object deprecated."""
    @wraps(obj)
    def wrapper(*args, **kwargs):
        name = '{}.{}'.format(obj.__self__.__class__, obj.__name__) if hasattr(obj, '__self__') else obj
        log.warning('Call to deprecated {}'.format(name))
        return obj(*args, **kwargs)
    return wrapper


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
        old__new__ = obj.__new__

        def _refuse__new__(cls, *args, **kwargs):
            if cls == obj:
                raise NotImplementedError("Can't instantiate abstract class " + obj.__name__)
            return old__new__(cls, *args, **kwargs)

        obj.__new__ = _refuse__new__
        return obj
    else:
        if not hasattr(obj, '__qualname__'):
            raise TypeError('Put @abstract decorator below (evaluated before) '
                            'any of @staticmethod, @classmethod, or @property.')
        cls_name = obj.__qualname__.rsplit('.', 1)[0]
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


# For best result, keep this at the bottom
__all__ = export_globals(globals(), __name__)

# ONLY NON-EXPORTED VALUES BELOW HERE
