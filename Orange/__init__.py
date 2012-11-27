from __future__ import absolute_import
from importlib import import_module

try:
    from .import version
    # Always use short_version here (see PEP 386)
    __version__ = version.short_version
    __hg_revision__ = version.hg_revision
except ImportError:
    __version__ = "unknown"
    __hg_revision__ = "unknown"

ADDONS_ENTRY_POINT = 'orange.addons'

import warnings
import pkg_resources

alreadyWarned = False
disabledMsg = "Some features will be disabled due to failing modules\n"


def _import(name):
    global alreadyWarned
    try:
        import_module(name, package='Orange')
    except ImportError as err:
        warnings.warn("%sImporting '%s' failed: %s" %
                      (disabledMsg if not alreadyWarned else "", name, err),
                      UserWarning, 2)
        alreadyWarned = True


_import(".data")

del _import
del alreadyWarned
del disabledMsg
