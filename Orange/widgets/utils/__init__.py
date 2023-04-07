import inspect
import sys
from collections import deque
from contextlib import contextmanager
from enum import Enum, IntEnum
from typing import (
    TypeVar, Callable, Any, Iterable, Optional, Hashable, Type, Union, Tuple
)
from xml.sax.saxutils import escape

from AnyQt.QtCore import QObject, Qt

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
    return {int(i * count / float(iterations)) for i in range(iterations)}


def getdeepattr(obj, attr, *arg, **kwarg):
    if isinstance(obj, dict):
        return obj.get(attr)
    return deepgetattr(obj, attr, *arg, **kwarg)


def to_html(s):
    return s.replace("<=", "≤").replace(">=", "≥"). \
        replace("<", "&lt;").replace(">", "&gt;").replace("=\\=", "≠")

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


def qname(type_: type) -> str:
    """Return the fully qualified name for a `type_`."""
    return "{0.__module__}.{0.__qualname__}".format(type_)


_T1 = TypeVar("_T1")  # pylint: disable=invalid-name
_E = TypeVar("_E", bound=Enum)  # pylint: disable=invalid-name
_A = TypeVar("_A")  # pylint: disable=invalid-name
_B = TypeVar("_B")  # pylint: disable=invalid-name


def apply_all(seq, op):
    # type: (Iterable[_T1], Callable[[_T1], Any]) -> None
    """Apply `op` on all elements of `seq`."""
    # from itertools recipes `consume`
    deque(map(op, seq), maxlen=0)


def ftry(
        func: Callable[..., _A],
        error: Union[Type[BaseException], Tuple[Type[BaseException]]],
        default: _B
) -> Callable[..., Union[_A, _B]]:
    """
    Wrap a `func` such that if `errors` occur `default` is returned instead.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except error:
            return default
    return wrapper


def unique_everseen(iterable, key=None):
    # type: (Iterable[_T1], Optional[Callable[[_T1], Hashable]]) -> Iterable[_T1]
    """
    Return an iterator over unique elements of `iterable` preserving order.

    If `key` is supplied it is used as a substitute for determining
    'uniqueness' of elements.

    Parameters
    ----------
    iterable : Iterable[T]
    key : Callable[[T], Hashable]

    Returns
    -------
    unique : Iterable[T]
    """
    seen = set()
    if key is None:
        key = lambda t: t
    for el in iterable:
        el_k = key(el)
        if el_k not in seen:
            seen.add(el_k)
            yield el


def enum_get(etype: Type[_E], name: str, default: _T1) -> Union[_E, _T1]:
    """
    Return an Enum member by `name`. If no such member exists in `etype`
    return `default`.
    """
    try:
        return etype[name]
    except LookupError:
        return default


def instance_tooltip(domain, row, skip_attrs=()):
    def show_part(_point_data, singular, plural, max_shown, _vars):
        cols = [escape('{} = {}'.format(var.name, _point_data[var]))
                for var in _vars[:max_shown + len(skip_attrs)]
                if _vars is domain.class_vars
                or var not in skip_attrs][:max_shown]
        if not cols:
            return ""
        n_vars = len(_vars)
        if n_vars > max_shown:
            cols[-1] = f'... and {n_vars - max_shown + 1} others'
        return \
            "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
            + "<br/>".join(cols)

    parts = (("Class", "Classes", 4, domain.class_vars),
             ("Meta", "Metas", 4, domain.metas),
             ("Feature", "Features", 10, domain.attributes))
    return "<br/>".join(show_part(row, *columns) for columns in parts)


def enum2int(enum: Union[Enum, IntEnum]) -> int:
    """
    PyQt5 uses IntEnum like object for settings, for example SortOrder while
    PyQt6 uses Enum. PyQt5's IntEnum also does not support value attribute.
    This function transform both settings objects to int.

    Parameters
    ----------
    enum
        IntEnum like object or Enum object with Qt's settings

    Returns
    -------
    Settings transformed to int
    """
    return int(enum) if isinstance(enum, int) else enum.value


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)
