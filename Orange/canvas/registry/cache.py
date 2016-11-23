"""
Widget Registry cache.

"""

import os
import pickle
import logging

from .. import config

log = logging.getLogger(__name__)


def registry_cache_filename():
    """Return the pickled registry cache filename. Also make sure the
    containing directory is created if it does not exists.

    """
    cache_dir = config.cache_dir()
    default = os.path.join(cache_dir, "registry-cache.pck")
    cache_filename = config.rc.get("registry.registry-cache", default)
    dirname = os.path.dirname(cache_filename)
    if not os.path.exists(dirname):
        log.info("Creating directory %r", dirname)
        os.makedirs(dirname)
    return cache_filename


def registry_cache():
    """Return the registry cache dictionary.
    """
    filename = registry_cache_filename()
    log.debug("Loading widget registry cache (%r).", filename)
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            log.error("Could not load registry cache.", exc_info=True)

    return {}


def save_registry_cache(cache):
    """Save (pickle) the registry cache. Return True on success,
    False otherwise.

    """
    filename = registry_cache_filename()
    log.debug("Saving widget registry cache with %i entries (%r).",
              len(cache), filename)
    try:
        with open(filename, "wb") as f:
            pickle.dump(cache, f)
        return True
    except Exception:
        log.error("Could not save registry cache", exc_info=True)
    return False
