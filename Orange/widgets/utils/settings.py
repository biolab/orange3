import typing
from typing import Any, Union, Tuple, Dict, List

from PyQt5.QtCore import QSettings

if typing.TYPE_CHECKING:  # pragma: no cover
    _T = typing.TypeVar("T")
    #: Specification for an value in the return value of readArray
    #: Can be single type or a tuple of (type, defaultValue) where default
    #: value is used where a stored entry is missing.
    ValueSpec = Union[
        typing.Type[_T], Tuple[typing.Type[_T], _T],
    ]


def QSettings_readArray(settings, key, scheme):
    # type: (QSettings, str, Dict[str, ValueSpec]) -> List[Dict[str, Any]]
    """
    Read the whole array from a QSettings instance.

    Parameters
    ----------
    settings : QSettings
    key : str
    scheme : Dict[str, ValueSpec]

    Example
    -------
    >>> s = QSettings("./login.ini")
    >>> QSettings_readArray(s, "array", {"username": str, "password": str})
    [{"username": "darkhelmet", "password": "1234"}}
    >>> QSettings_readArray(
    ...    s, "array", {"username": str, "noexist": (str, "~||~")})
    ...
    [{"username": "darkhelmet", "noexist": "~||~"}}
    """
    items = []
    count = settings.beginReadArray(key)

    def normalize_spec(spec):
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError("len(spec) != 2")
            type_, default = spec
        else:
            type_, default = spec, None
        return type_, default

    specs = {
        name: normalize_spec(spec) for name, spec in scheme.items()
    }
    for i in range(count):
        settings.setArrayIndex(i)
        item = {}
        for key, (type_, default) in specs.items():
            value = settings.value(key, type=type_, defaultValue=default)
            item[key] = value
        items.append(item)
    settings.endArray()
    return items


def QSettings_writeArray(settings, key, values):
    # type: (QSettings, str, List[Dict[str, Any]]) -> None
    """
    Write an array of values to a QSettings instance.

    Parameters
    ----------
    settings : QSettings
    key : str
    values : List[Dict[str, Any]]

    Examples
    --------
    >>> s = QSettings("./login.ini")
    >>> QSettings_writeArray(
    ...     s, "array", [{"username": "darkhelmet", "password": "1234"}]
    ... )
    """
    settings.beginWriteArray(key, len(values))
    for i in range(len(values)):
        settings.setArrayIndex(i)
        for key_, val in values[i].items():
            settings.setValue(key_, val)
    settings.endArray()


def QSettings_writeArrayItem(settings, key, index, item, arraysize):
    # type: (QSettings, str, int, Dict[str, Any], int) -> None
    """
    Write/update an array item at index.

    Parameters
    ----------
    settings : QSettings
    key : str
    index : int
    item : Dict[str, Any]
    arraysize : int
        The full array size. Note that the array will be truncated to this
        size.
    """
    settings.beginWriteArray(key, arraysize)
    settings.setArrayIndex(index)
    for key_, val in item.items():
        settings.setValue(key_, val)
    settings.endArray()
