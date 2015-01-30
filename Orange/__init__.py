try:
    from .import version
    # Always use short_version here (see PEP 386)
    __version__ = version.short_version
    __git_revision__ = version.git_revision
except ImportError:
    __version__ = "unknown"
    __git_revision__ = "unknown"

ADDONS_ENTRY_POINT = 'orange.addons'


already_warned = False
disabled_msg = "Some features will be disabled due to failing modules\n"


def _import(name):
    from importlib import import_module
    import warnings
    global already_warned
    try:
        import_module(name, package='Orange')
    except ImportError as err:
        warnings.warn("%sImporting '%s' failed: %s" %
                      (disabled_msg if not already_warned else "", name, err),
                      UserWarning, 2)
        already_warned = True


# noinspection PyUnresolvedReferences
def import_all():
    from importlib import import_module
    import Orange
    for name in ["classification", "clustering", "data", "distance",
                 "evaluation", "feature", "misc", "regression", "statistics"]:
        setattr(Orange, name, import_module('Orange.' + name, package='Orange'))

_import(".data")

class LazyModule:
    def __init__(self, name):
        self.__name = name

    def do_import(self):
        import Orange
        mod = import_module('Orange.' + self.__name, package='Orange')
        setattr(Orange, self.__name, mod)
        return mod

    def __getattr__(self, key):
        return getattr(self.do_import(), key)

    def _getAttributeNames(self):
        return list(self.do_import().__dict__)

classification = LazyModule('classification')

del _import
del already_warned
del disabled_msg
