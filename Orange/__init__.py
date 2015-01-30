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
        Orange.__dict__[name] = import_module('Orange.' + name, package='Orange')

    # Alternatives:
    #     global classification
    #     import Orange.classification as classification
    # or
    #     import Orange.classification as classification
    #     globals()['clasification'] = classification

_import(".data")

del _import
del already_warned
del disabled_msg
