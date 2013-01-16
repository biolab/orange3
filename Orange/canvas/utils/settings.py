"""
Settings (`settings`)
=====================

A more `dict` like interface for QSettings

"""

import abc
import logging

from collections import namedtuple, MutableMapping

from PyQt4.QtCore import QObject, QVariant, QString, QChar, QEvent, \
                         QSettings, QCoreApplication

from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import PYQT_VERSION, pyqtWrapperType

from . import toPyObject

log = logging.getLogger(__name__)


config_slot = namedtuple(
    "config_slot",
    ["key",
     "value_type",
     "default_value",
     "doc"]
)


class SettingChangedEvent(QEvent):
    """
    A settings has changed.

    This event is sent by Settings instance to itself when a setting for
    a key has changed.

    """
    SettingChanged = QEvent.registerEventType()
    """Setting was changed"""

    SettingAdded = QEvent.registerEventType()
    """A setting was added"""

    SettingRemoved = QEvent.registerEventType()
    """A setting was removed"""

    def __init__(self, etype, key, value=None, oldValue=None):
        """
        Initialize the event instance
        """
        QEvent.__init__(self, etype)
        self.__key = key
        self.__value = value
        self.__oldValue = oldValue

    def key(self):
        return self.__key

    def value(self):
        return self.__value

    def oldValue(self):
        return self.__oldValue


class _Settings(QSettings):
    def __init__(self, parent=None):
        QSettings.__init__(self, parent)

    def _fullKey(self, key):
        group = unicode(self.group())
        if group:
            return "{0}/{1}".format(group.rstrip("/"), unicode(key))
        else:
            return unicode(key)

    def _value(self, key, defaultValue, type):
        if PYQT_VERSION < 0x40803:
            # QSettings.value does not have `type` argument before PyQt4 4.8.3
            value = QSettings.value(self, key, defaultValue)

            if type is not None:
                value = qvariant_to_py(value, type)
        else:
            value = QSettings.value(self, key, defaultValue, type)

        return value

    def value(self, key, defaultValue=None, type=None):
        """
        """
        if QSettings.contains(self, key):
            value = self._value(key, defaultValue, type)
        else:
            value = QSettings.value(self, key, defaultValue)
        return value


def _check_error((val, status)):
    if not status:
        raise TypeError()
    else:
        return val


def qvariant_to_py(variant, py_type):
    """Convert a QVariant object to a python object of type `py_type`.
    """
    vtype = variant.type()

    if vtype == QVariant.Invalid:
        if py_type is not None:
            raise TypeError("Invalid variant type")
        else:
            return None

    elif vtype in [QVariant.Hash, QVariant.Map]:
        variant = variant.toPyObject()
        return dict((unicode(key), py_type(value))
                    for key, value in variant.items())

    elif vtype == QVariant.List:
        variant = variant.toPyObject()
        return [py_type(item) for item in variant]

    if issubclass(py_type, basestring):
        return py_type(variant.toString())
    elif py_type == bool:
        return variant.toBool()
    elif py_type == int:
        return _check_error(variant.toInt())
    elif py_type == float:
        return _check_error(variant.toDouble())
    else:
        raise NotImplementedError


def qt_to_mapped_type(value):
    """Try to convert a Qt value to the corresponding python
    mapped type (i.e. QString to unicode, etc.)

    """
    if isinstance(value, QString):
        return unicode(value)
    elif isinstance(value, QChar):
        return str(value)
    else:
        return value


class QABCMeta(pyqtWrapperType, abc.ABCMeta):
    def __init__(self, name, bases, attr_dict):
        pyqtWrapperType.__init__(self, name, bases, attr_dict)
        abc.ABCMeta.__init__(self, name, bases, attr_dict)


class _pickledvalue(object):
    def __init__(self, value):
        self.value = value


class Settings(QObject, MutableMapping):
    __metaclass__ = QABCMeta

    valueChanged = Signal(unicode, object)
    valueAdded = Signal(unicode, object)
    keyRemoved = Signal(unicode)

    def __init__(self, parent=None, defaults=(), path=None, store=None):
        QObject.__init__(self, parent)

        if store is None:
            store = QSettings()

        path = path = (path or "").rstrip("/")

        self.__path = path
        self.__defaults = dict([(slot.key, slot) for slot in defaults])
        self.__store = store

    def __key(self, key):
        if self.__path:
            return "/".join([self.__path, key])
        else:
            return key

    def __delitem__(self, key):
        """
        Delete the setting for key. If key is a group remove the
        whole group.

        .. note:: defaults cannot be deleted they are instead reverted
                  to their original state.

        """
        if key not in self:
            raise KeyError(key)

        if self.isgroup(key):
            group = self.group(key)
            for key in group:
                del group[key]

        else:
            fullkey = self.__key(key)

            oldValue = self.get(key)

            if self.__store.contains(fullkey):
                self.__store.remove(fullkey)

            newValue = None
            if fullkey in self.__defaults:
                newValue = self.__defaults[fullkey].default_value
                etype = SettingChangedEvent.SettingChanged
            else:
                etype = SettingChangedEvent.SettingRemoved

            QCoreApplication.sendEvent(
                self, SettingChangedEvent(etype, key, newValue, oldValue)
            )

    def __value(self, fullkey, value_type):
        typesafe = value_type is not None
        if PYQT_VERSION < 0x40803:
            # QSettings.value does not have `type` argument.
            value = toPyObject(self.__store.value(fullkey))
            typesafe = False
        else:
            if value_type is None:
                value = toPyObject(self.__store.value(fullkey))
            else:
                try:
                    value = self.__store.value(fullkey, type=value_type)
                except TypeError:
                    # In case the value was pickled in a type unsafe mode
                    value = toPyObject(self.__store.value(fullkey))
                    typesafe = False

        if not typesafe:
            if isinstance(value, _pickledvalue):
                value = value.value
            else:
                log.warning("value for key %r is not a '_pickledvalue' (%r),"
                            "possible loss of type information.",
                            fullkey,
                            type(value))

        return value

    def __setValue(self, fullkey, value, value_type=None):
        typesafe = value_type is not None
        if PYQT_VERSION < 0x40803:
            # QSettings.value does not have `type` argument.
            typesafe = False

        if not typesafe:
            # value is stored in a _pickledvalue wrapper to force PyQt
            # to store it in a pickled format so we don't lose the type
            # TODO: Could check if QSettings.Format stores type info.
            value = _pickledvalue(value)

        self.__store.setValue(fullkey, value)

    def __getitem__(self, key):
        """
        Get the setting for key.
        """
        if key not in self:
            raise KeyError(key)

        if self.isgroup(key):
            raise KeyError("is a group")

        fullkey = self.__key(key)
        slot = self.__defaults.get(fullkey, None)

        if self.__store.contains(fullkey):
            value = self.__value(fullkey, slot.value_type if slot else None)
        else:
            value = slot.default_value

        return value

    def __setitem__(self, key, value):
        """
        Set the setting for key.
        """
        if not isinstance(key, basestring):
            raise TypeError(key)

        fullkey = self.__key(key)
        value_type = None

        if fullkey in self.__defaults:
            value_type = self.__defaults[fullkey].value_type
            if not isinstance(value, value_type):
                value = qt_to_mapped_type(value)
                if not isinstance(value, value_type):
                    raise TypeError("Expected {0!r} got {1!r}".format(
                                        value_type.__name__,
                                        type(value).__name__)
                                    )

        if key in self:
            oldValue = self.get(key)
            etype = SettingChangedEvent.SettingChanged
        else:
            oldValue = None
            etype = SettingChangedEvent.SettingAdded

        self.__setValue(fullkey, value, value_type)

        QCoreApplication.sendEvent(
            self, SettingChangedEvent(etype, key, value, oldValue)
        )

    def __contains__(self, key):
        """
        Return `True` if settings contain the `key`, False otherwise.
        """
        fullkey = self.__key(key)
        return self.__store.contains(fullkey) or (fullkey in self.__defaults)

    def __iter__(self):
        """Return an iterator over over all keys.
        """
        keys = map(unicode, self.__store.allKeys()) + \
               self.__defaults.keys()

        if self.__path:
            path = self.__path + "/"
            keys = filter(lambda key: key.startswith(path), keys)
            keys = [key[len(path):] for key in keys]

        return iter(sorted(set(keys)))

    def __len__(self):
        return len(list(iter(self)))

    def group(self, path):
        if self.__path:
            path = "/".join([self.__path, path])

        return Settings(self, self.__defaults.values(), path, self.__store)

    def isgroup(self, key):
        """Is the `key` a settings group i.e. does it have subkeys.
        """
        if key not in self:
            raise KeyError("{0!r} is not a valid key".format(key))

        return len(self.group(key)) > 0

    def isdefault(self, key):
        """Is the value for key the default.
        """
        if key not in self:
            raise KeyError(key)
        return not self.__store.contains(self.__key(key))

    def clear(self):
        """Clear the settings and restore the defaults.
        """
        self.__store.clear()

    def add_default_slot(self, default):
        """Add a default slot to the settings This also replaces any
        previously set value for the key.

        """
        value = default.default_value
        oldValue = None
        etype = SettingChangedEvent.SettingAdded
        key = default.key

        if key in self:
            oldValue = self.get(key)
            etype = SettingChangedEvent.SettingChanged
            if not self.isdefault(key):
                # Replacing a default value.
                self.__store.remove(self.__key(key))

        self.__defaults[key] = default
        event = SettingChangedEvent(etype, key, value, oldValue)
        QCoreApplication.sendEvent(self, event)

    def get_default_slot(self, key):
        return self.__defaults[self.__key(key)]

    def values(self):
        """Return a list over of all values in the settings.
        """
        return MutableMapping.values(self)

    def customEvent(self, event):
        QObject.customEvent(self, event)

        if isinstance(event, SettingChangedEvent):
            if event.type() == SettingChangedEvent.SettingChanged:
                self.valueChanged.emit(event.key(), event.value())
            elif event.type() == SettingChangedEvent.SettingAdded:
                self.valueAdded.emit(event.key(), event.value())
            elif event.type() == SettingChangedEvent.SettingRemoved:
                self.keyRemoved.emit(event.key())

            parent = self.parent()
            if isinstance(parent, Settings):
                # Assumption that the parent is a parent setting group.
                parent.customEvent(
                    SettingChangedEvent(event.type(),
                                        "/".join([self.__path, event.key()]),
                                        event.value(),
                                        event.oldValue())
                )
