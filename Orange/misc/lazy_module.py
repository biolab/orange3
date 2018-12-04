class _LazyModule:
    def __init__(self, name):
        self.__name = name

    def _do_import(self):
        import Orange
        from importlib import import_module
        mod = import_module('Orange.' + self.__name, package='Orange')
        setattr(Orange, self.__name, mod)
        return mod

    def __getattr__(self, key):
        return getattr(self._do_import(), key)

    def __dir__(self):
        return list(self._do_import().__dict__)
