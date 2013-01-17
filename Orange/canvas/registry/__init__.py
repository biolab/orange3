"""
===============
Widget Registry
===============

A registry (and discovery) of available widgets.

"""

import logging

# Color names that can be used in widget/category descriptions
# as background color.
NAMED_COLORS = \
    {"light-orange": "#FFD39F",
     "orange": "#FFA840",
     "light-red": "#FFB7B1",
     "red": "#FF7063",
     "light-pink": "#FAC1D9",
     "pink": "#F584B4",
     "light-purple": "#E5BBFB",
     "purple": "#CB77F7",
     "light-blue": "#CAE1FC",
     "blue": "#95C3F9",
     "light-turquoise": "#C3F3F3",
     "turquoise": "#87E8E8",
     "light-green": "#ACE3CE",
     "green": "#5AC79E",
     "light-grass": "#DFECB0",
     "grass": "#C0D962",
     "light-yellow": "#F7F5A7",
     "yellow": "#F0EC4F",
     }


from .description import (
    WidgetDescription, CategoryDescription,
    InputSignal, OutputSignal
)

from .base import WidgetRegistry, VERSION_HEX

log = logging.getLogger(__name__)


__GLOBAL_REGISTRY = None


def global_registry():
    """Return a global WidgetRegistry instance.
    """
    global __GLOBAL_REGISTRY
    # TODO: lock
    if __GLOBAL_REGISTRY is None:
        log.debug("'global_registry()' - running widget discovery.")
        from . import discovery
        reg = WidgetRegistry()
        disc = discovery.WidgetDiscovery(reg)
        disc.run()
        log.info("'global_registry()' discovery finished.")
        __GLOBAL_REGISTRY = reg

    return __GLOBAL_REGISTRY


def set_global_registry(registry):
    global __GLOBAL_REGISTRY
    log.debug("'set_global_registry()' - setting registry.")
    __GLOBAL_REGISTRY = registry
