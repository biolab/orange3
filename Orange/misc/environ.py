import os
import sys

import Orange


def widget_settings_dir():
    """
    Return the platform dependent directory where widgets save their settings.
    """
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform == "win32":
        base = os.getenv("APPDATA", os.path.expanduser("~/AppData/Local"))
    elif os.name == "posix":
        base = os.getenv('XDG_DATA_HOME', os.path.expanduser("~/.local/share"))
    else:
        base = os.path.expanduser("~/.local/share")

    return os.path.join(base, "Orange Canvas", Orange.__version__, "widgets")


def cache_dir():
    """
    Return the platform dependent cache directory.
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
        return os.path.join(base, "Cache")
    else:
        return base
