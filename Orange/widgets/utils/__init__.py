import inspect
import sys

from AnyQt.QtCore import QObject

from Orange.data.variable import TimeVariable
from Orange.util import deepgetattr


def vartype(var):
    if var.is_discrete:
        return 1
    elif var.is_continuous:
        if isinstance(var, TimeVariable):
            return 4
        return 2
    elif var.is_string:
        return 3
    else:
        return 0


def progress_bar_milestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])


def getdeepattr(obj, attr, *arg, **kwarg):
    if isinstance(obj, dict):
        return obj.get(attr)
    return deepgetattr(obj, attr, *arg, **kwarg)


def to_html(str):
    return str.replace("<=", "&#8804;").replace(">=", "&#8805;").\
        replace("<", "&#60;").replace(">", "&#62;").replace("=\\=", "&#8800;")

getHtmlCompatibleString = to_html


def get_variable_values_sorted(variable):
    """
    Return a list of sorted values for given attribute, if all its values can be
    cast to int's.
    """
    if variable.is_continuous:
        return []
    try:
        return sorted(variable.values, key=int)
    except ValueError:
        return variable.values


def dumpObjectTree(obj, _indent=0):
    """
    Dumps Qt QObject tree. Aids in debugging internals.
    See also: QObject.dumpObjectTree()
    """
    assert isinstance(obj, QObject)
    print('{indent}{type} "{name}"'.format(indent=' ' * (_indent * 4),
                                           type=type(obj).__name__,
                                           name=obj.objectName()),
          file=sys.stderr)
    for child in obj.children():
        dumpObjectTree(child, _indent + 1)


def getmembers(obj, predicate=None):
    """Return all the members of an object in a list of (name, value) pairs sorted by name.

    Behaves like inspect.getmembers. If a type object is passed as a predicate,
    only members of that type are returned.
    """

    if isinstance(predicate, type):
        def mypredicate(x):
            return isinstance(x, predicate)
    else:
        mypredicate = predicate
    return inspect.getmembers(obj, mypredicate)
