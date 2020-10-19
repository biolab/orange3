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
