from collections import defaultdict

def getCached(data, funct, params = (), kwparams = {}):
    if not data: return None
    if getattr(data, "info", None) == None or data.info["__version__"] != data.version: 
        setattr(data, "info", {"__version__": data.version})

    if data.info.has_key(funct):
        return data.info[funct]
    else:
        if type(funct) != str:
            data.info[funct] = funct(*params, **kwparams)
            return data.info[funct]
        else:
            return None 
         

def setCached(data, name, info):
    if not data: return
    if getattr(data, "info", None) == None or data.info["__version__"] != data.version:
        setattr(data, "info", {"__version__": data.version})
    data.info[name] = info

def delCached(data, name):
    if not data: return
    if getattr(data, "info", None) != None and data.info.has_key(name):
        del data.info[name]
        
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
        return sorted(weighted_hints.items(), key=lambda key:key[1], reverse=True)
    
    def get_hint(self, data, key, default=None):
        hints = self.get_weighted_hints(data, key)
        if hints:
            return hints[0][0]
        else:
            return default
        
data_hints = DataHintsCache()
    