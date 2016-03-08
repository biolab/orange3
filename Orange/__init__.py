from .misc.lazy_module import _LazyModule
from .misc.datasets import _DatasetInfo
from .version import \
    short_version as __version__, git_revision as __git_version__

ADDONS_ENTRY_POINT = 'orange.addons'

from Orange import data

for mod_name in ['classification', 'clustering', 'distance', 'ensembles',
                 'evaluation', 'misc', 'preprocess', 'projection', 'regression',
                 'statistics', 'version', 'widgets']:
    globals()[mod_name] = _LazyModule(mod_name)

datasets = _DatasetInfo()

del mod_name
