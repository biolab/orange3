from .misc.lazy_module import LazyModule
from .version import \
    short_version as __version__, git_revision as __git_version__

ADDONS_ENTRY_POINT = 'orange.addons'

from Orange import data

for mod_name in ['classification', 'clustering', 'distance', 'evaluation',
                 'feature', 'misc', 'regression', 'statistics', 'widgets']:
    globals()[mod_name] = LazyModule(mod_name)

del mod_name
del LazyModule