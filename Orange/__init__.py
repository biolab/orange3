from .misc.lazy_module import _LazyModule
from .version import \
    short_version as __version__, git_revision as __git_version__

ADDONS_ENTRY_POINT = 'orange.addons'

from Orange import data

for mod_name in ['classification', 'clustering', 'distance', 'evaluation',
                 'misc', 'preprocess', 'projection', 'regression',
                 'statistics', 'widgets']:
    globals()[mod_name] = _LazyModule(mod_name)

del mod_name
