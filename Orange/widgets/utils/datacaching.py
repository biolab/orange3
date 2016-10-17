from collections import defaultdict
from operator import itemgetter

def getCached(data, funct, params=(), **kwparams):
    # pylint: disable=protected-access
    if data is None:
        return None
    if not hasattr(data, "__data_cache"):
        data.__data_cache = {}
    info = data.__data_cache
    if funct in info:
        return info[funct]
    if isinstance(funct, str):
        return None
    info[funct] = res = funct(*params, **kwparams)
    return res


def setCached(data, name, value):
    if data is None:
        return
    if not hasattr(data, "__data_cache"):
        data.__data_cache = {}
    data.__data_cache[name] = value

def delCached(data, name):
    info = data is not None and getattr(data, "__data_cache")
    if info and name in info:
        del info[name]


class DataHintsCache(object):
    def __init__(self, ):
        self._hints = defaultdict(lambda: defaultdict(list))
        pass

    def set_hint(self, data, key, value, weight=1.0):
        attrs = data.domain.variables + data.domain.metas
        for attr in attrs:
            self._hints[key][attr].append((value, weight/len(attrs)))

    def get_weighted_hints(self, data, key):
        attrs = data.domain.variables + data.domain.metas
        weighted_hints = defaultdict(float)
        for attr in attrs:
            for val, w in self._hints[key][attr]:
                weighted_hints[val] += w
        return sorted(weighted_hints.items(), key=itemgetter(1), reverse=True)

    def get_hint(self, data, key, default=None):
        hints = self.get_weighted_hints(data, key)
        if hints:
            return hints[0][0]
        else:
            return default

data_hints = DataHintsCache()
