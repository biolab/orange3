"""
Utility functions.

"""


def name_lookup(qualified_name, globals={}):
    """Return the object referenced by a qualified name (doted name).
    """
    module_name, class_name = qualified_name.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name], globals=globals)
    return getattr(module, class_name)


def qualified_name(qualified_object):
    """Return a qualifeid name for `qualified_obj` (type or function).
    """
    return "%s.%s" % (qualified_object.__module__, qualified_object.__name__)


def check_type(obj, type_or_tuple):
    if not isinstance(obj, type_or_tuple):
        raise TypeError("Expected %r. Got %r" % (type_or_tuple, type(obj)))


def check_subclass(cls, class_or_tuple):
    if not issubclass(cls, class_or_tuple):
        raise TypeError("Expected %r. Got %r" % (class_or_tuple, type(cls)))


def check_arg(pred, value):
    if not pred:
        raise ValueError(value)
