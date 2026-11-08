# This module is a mixture of imports and code, so we allow import anywhere
# pylint: disable=wrong-import-position,wrong-import-order

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

# A hack that prevents segmentation fault with Nvidia drives on Linux if Qt's browser window
# is shown (seen in https://github.com/spyder-ide/spyder/pull/7029/files)
try:
    import ctypes
    ctypes.CDLL("libGL.so.1", mode=ctypes.RTLD_GLOBAL)
except:  # pylint: disable=bare-except
    pass
finally:
    del ctypes
