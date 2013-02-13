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
from . import discovery

log = logging.getLogger(__name__)


__GLOBAL_REGISTRY = {}


def global_registry(entry_point_group="_default"):
    """
    Return a global WidgetRegistry instance for the entry point group.
    If none exists then it will be created.

    .. note:: This will be deprecated when a proper replacement for it's
              uses can be found.

    """
    global __GLOBAL_REGISTRY
    # TODO: lock
    if __GLOBAL_REGISTRY.get(entry_point_group) is None:
        log.debug("'global_registry()' - running widget discovery.")
        if entry_point_group == "_default":
            from ..config import widgets_entry_points
            entry_points_iter = widgets_entry_points()
        else:
            entry_points_iter = entry_point_group
        reg = WidgetRegistry()
        disc = discovery.WidgetDiscovery(reg)
        disc.run(entry_points_iter)
        log.info("'global_registry()' discovery finished.")
        __GLOBAL_REGISTRY[entry_point_group] = reg

    return __GLOBAL_REGISTRY[entry_point_group]


def set_global_registry(registry, entry_point_group="_default"):
    """
    Set the global WidgetRegistry instance for the entry point group.

    .. note:: Overrides previous registry.

    """
    global __GLOBAL_REGISTRY
    log.debug("'set_global_registry()' - setting registry.")
    __GLOBAL_REGISTRY[entry_point_group] = registry
