"""
environ
=======

This module contains some basic configuration options for Orange
(for now mostly changing directories where settings and data are saved).

How it works
------------

The configuration is read from '{sys.prefix}/etc/orangerc.cfg'
which is a standard `configparser` file.

orangerc.cfg
------------

.. code-block:: cfg

    # An exemple orangerc.cfg file
    # ----------------------------
    #
    # A number of variables are predefined:
    # - prefix: `sys.prefix`
    # - name: The application/library name ('Orange')
    # - version: The application/library name ('Orange.__version__')
    # - version.major, version.minor, version.micro: The version components

    [paths]
    # The base path where persistent application data can be stored
    # (here we define a prefix relative path)
    data_dir_base = %(prefix)s/share
    # the base path where application data can be stored
    cache_dir = %(prefix)s/cache/%(name)s/%(version)s

    # The following is only applicable for a running orange canvas application.

    # The base dir where widgets store their settings
    widget_settings_dir = %(prefix)s/config/%(name)s/widgets
    # The base dir where canvas stores its settings
    canvas_settings_dir = %(prefix)s/config/%(name)s/canvas

"""
import os
import sys
import warnings
import sysconfig
import configparser

from typing import Optional

import Orange


def _get_parsed_config():
    version = Orange.__version__.split(".")
    data = sysconfig.get_path("data")
    vars = {
        "home": os.path.expanduser("~/"),
        "prefix": sys.prefix,
        "data": sysconfig.get_path("data"),
        "name": "Orange",
        "version": Orange.__version__,
        "version.major": version[0],
        "version.minor": version[1],
        "version.micro": version[2],
    }
    conf = configparser.ConfigParser(vars)
    conf.read([
        os.path.join(data, "etc/orangerc.conf"),
    ], encoding="utf-8")
    if not conf.has_section("paths"):
        conf.add_section("paths")
    return conf


def get_path(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get configured path

    Parameters
    ----------
    name: str
        The named config path value
    default: Optional[str]
        The default to return if `name` is not defined
    """
    cfg = _get_parsed_config()
    try:
        return cfg.get('paths', name)
    except (configparser.NoOptionError, configparser.NoSectionError):
        return default


def _default_data_dir_base():
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform == "win32":
        base = os.getenv("APPDATA", os.path.expanduser("~/AppData/Local"))
    elif os.name == "posix":
        base = os.getenv('XDG_DATA_HOME', os.path.expanduser("~/.local/share"))
    else:
        base = os.path.expanduser("~/.local/share")
    return base


def data_dir_base():
    """
    Return the platform dependent application directory.

    This is usually

        - on windows: "%USERPROFILE%\\AppData\\Local\\"
        - on OSX:  "~/Library/Application Support/"
        - other: "~/.local/share/
    """
    return get_path('data_dir_base', _default_data_dir_base())


def data_dir(versioned=True):
    """
    Return the platform dependent Orange data directory.

    This is ``data_dir_base()``/Orange/__VERSION__/ directory if versioned is
    `True` and ``data_dir_base()``/Orange/ otherwise.
    """
    base = data_dir_base()
    if versioned:
        return os.path.join(base, "Orange", Orange.__version__)
    else:
        return os.path.join(base, "Orange")


def widget_settings_dir(versioned=True):
    """
    Return the platform dependent directory where widgets save their settings.

    .. deprecated:: 3.23
    """
    warnings.warn(
        f"'{__name__}.widget_settings_dir' is deprecated.",
        DeprecationWarning, stacklevel=2
    )
    import orangewidget.settings
    return orangewidget.settings.widget_settings_dir(versioned)


def _default_cache_dir():
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif sys.platform == "win32":
        base = os.getenv("APPDATA", os.path.expanduser("~/AppData/Local"))
    elif os.name == "posix":
        base = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    else:
        base = os.path.expanduser("~/.cache")

    base = os.path.join(base, "Orange", Orange.__version__)
    if sys.platform == "win32":
        # On Windows cache and data dir are the same.
        # Microsoft suggest using a Cache subdirectory
        return os.path.join(base, "Cache")
    else:
        return base


def cache_dir(*args):
    """
    Return the platform dependent Orange cache directory.
    """
    return get_path("cache_dir", _default_cache_dir())
