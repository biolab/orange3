"""Various small utilities that might be useful everywhere"""

from itertools import chain
import numpy as np


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
        cls_name = obj.__qualname__.rsplit('.', 1)[0]
        def _refuse__call__(*args, **kwargs):
            raise NotImplementedError("Can't call abstract method {} of class {}"
                                      .format(obj.__name__, cls_name))
        return _refuse__call__


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
