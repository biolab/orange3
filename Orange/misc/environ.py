"""
Retrive basic library/application data/cache locations.

The basic FS layout for Orange data files is

$DATA_HOME/Orange/$VERSION/
    widgets/
    canvas/

where DATA_HOME is a platform dependent application directory
(:ref:`data_dir_base`) and VERSION is Orange.__version__ string.

``canvas`` subdirectory is reserved for settings/preferences stored
by Orange Canvas
``widget`` subdirectory is reserved for settings/preferences stored
by OWWidget

"""

import os
import sys

import Orange


def data_dir_base():
    """
    Return the platform dependent application directory.

    This is usually

        - on windows: "%USERPROFILE%\\AppData\\Local\\"
        - on OSX:  "~/Library/Application Support/"
        - other: "~/.local/share/
    """

    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform == "win32":
        base = os.getenv("APPDATA", os.path.expanduser("~/AppData/Local"))
    elif os.name == "posix":
        base = os.getenv('XDG_DATA_HOME', os.path.expanduser("~/.local/share"))
    else:
        base = os.path.expanduser("~/.local/share")
    return base


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

    This a subdirectory of ``data_dir(versioned)`` named "widgets"
    """
    return os.path.join(data_dir(versioned=versioned), "widgets")


def cache_dir(*args):
    """
    Return the platform dependent Orange cache directory.
    """
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
