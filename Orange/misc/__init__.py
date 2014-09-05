from importlib import import_module

def import_late_warning(name):
    try:
        return import_module(name)
    except ImportError:
        class Warn:
            def __getattr__(self, val):
                raise ImportError("Please install package '" + name + "' to use this functionality.")
        return Warn()
