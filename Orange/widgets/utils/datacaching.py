from collections import defaultdict
from operator import itemgetter

def getCached(data, funct, params=(), kwparams=None):
    if data is None:
        return None
    info = getattr(data, "__data_cache", None)
    if info is not None:
        if funct in data.info:
            return data.info(funct)
    else:
        info = data.info = {}
    if isinstance(funct, str):
        return None
    if kwparams is None:
        kwparams = {}
    info[funct] = res = funct(*params, **kwparams)
    return res


def setCached(data, name, value):
    if data is None:
        return
    info = getattr(data, "__data_cache")
    if info is None:
        info = data.info = {}
    info[name] = value

def delCached(data, name):
    info = data is not None and getattr(data, "__data_cache")
    if info and name in info:
        del info[name]


class DataHintsCache(object):
    def __init__(self, ):
        self._hints = defaultdict(lambda: defaultdict(list))
        pass

    def set_hint(self, data, key, value, weight=1.0):
        attrs = data.domain.variables + data.domain.getmetas().values()
        for attr in attrs:
            self._hints[key][attr].append((value, weight/len(attrs)))

    def get_weighted_hints(self, data, key):
        attrs = data.domain.variables + data.domain.getmetas().values()
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
