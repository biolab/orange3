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

