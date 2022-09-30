import re
from typing import List, Iterable


class frozendict(dict):
    def clear(self):
        raise AttributeError("FrozenDict does not support method 'clear'")

    def pop(self, _k):
        raise AttributeError("FrozenDict does not support method 'pop'")

    def popitem(self):
        raise AttributeError("FrozenDict does not support method 'popitem'")

    def setdefault(self, _k, _v):
        raise AttributeError("FrozenDict does not support method 'setdefault'")

    def update(self, _d):
        raise AttributeError("FrozenDict does not support method 'update'")

    def __setitem__(self, _key, _value):
        raise AttributeError("FrozenDict does not allow setting elements")

    def __delitem__(self, _key):
        raise AttributeError("FrozenDict does not allow deleting elements")


def natural_sorted(values: Iterable) -> List:
    """
    Sort values with natural sort or human order - [sth1, sth2, sth10] or
    [1, 2, 10]

    Parameters
    ----------
    values
        List with values to sort

    Returns
    -------
    List with sorted values
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(element):
        """
        alist.sort(key=natural_keys) or sorted(alist, key=natural_keys) sorts
        in human order
        """
        if isinstance(element, (str, bytes)):
            return [atoi(c) for c in re.split(r'(\d+)', element)]
        else:
            return element

    return sorted(values, key=natural_keys)


class DictMissingConst(dict):
    """
    `dict` with a constant for `__missing__()` value.

    This is mostly used for speed optimizations where
    `DictMissingConst(default, d).__getitem__(k)` is the least overhead
    equivalent to `d.get(k, default)` in the case where misses are not
    frequent by avoiding LOAD_* bytecode instructions for `default` at
    every call.

    Note
    ----
    This differs from `defaultdict(lambda: CONST)` in that misses do not
    grow the dict.

    Parameters
    ----------
    missing: Any
        The missing constant
    *args
    **kwargs
        The `*args`, and `**kwargs` are passed to `dict` constructor.
    """
    __slots__ = ("__missing",)

    def __init__(self, missing, *args, **kwargs):
        self.__missing = missing
        super().__init__(*args, **kwargs)

    @property
    def missing(self):
        return self.__missing

    def __missing__(self, key):
        return self.__missing

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, DictMissingConst) \
            and self.missing == other.missing

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce_ex__(self, protocol):
        return type(self), (self.missing, list(self.items())), \
               getattr(self, "__dict__", None)

    def copy(self):
        return type(self)(self.missing, self)

    def __repr__(self):
        return f"{type(self).__qualname__}({self.missing!r}, {dict(self)!r})"
