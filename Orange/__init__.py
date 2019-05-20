# This module is a mixture of imports and code, so we allow import anywhere
# pylint: disable=wrong-import-position,wrong-import-order

try:
    from Orange.data import _variable
except ImportError:
    raise ImportError("Compiled libraries cannot be found.\n"
                      "Try reinstalling the package with:\n"
                      "pip install --no-binary Orange3") from None

from Orange import data

from .misc.lazy_module import _LazyModule
from .misc.datasets import _DatasetInfo
from .version import \
    short_version as __version__, git_revision as __git_version__

ADDONS_ENTRY_POINT = 'orange.addons'

for mod_name in ['classification', 'clustering', 'distance', 'ensembles',
                 'evaluation', 'misc', 'modelling', 'preprocess', 'projection',
                 'regression', 'statistics', 'version', 'widgets']:
    globals()[mod_name] = _LazyModule(mod_name)

datasets = _DatasetInfo()

del mod_name

# If Qt is available (GUI) and Qt5, install backport for PyQt4 imports
try:
    import AnyQt.importhooks
except ImportError:
    pass
else:
    if AnyQt.USED_API == "pyqt5":
        # Make the chosen PyQt version pinned
        from AnyQt.QtCore import QObject
        del QObject

        import pyqtgraph  # import pyqtgraph first so that it can detect Qt5
        del pyqtgraph

        AnyQt.importhooks.install_backport_hook('pyqt4')
    del AnyQt


# A hack that prevents segmentation fault with Nvidia drives on Linux if Qt's browser window
# is shown (seen in https://github.com/spyder-ide/spyder/pull/7029/files)
try:
    import ctypes
    ctypes.CDLL("libGL.so.1", mode=ctypes.RTLD_GLOBAL)
except:  # pylint: disable=bare-except
    pass
finally:
    del ctypes
