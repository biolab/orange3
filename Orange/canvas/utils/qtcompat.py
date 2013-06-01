"""
PyQt4 compatibility utility functions.

.. warning:: It is important that any `sip.setapi` (at least for QVariant
             and QString) calls are already made before importing this
             module.

"""
from operator import methodcaller

import sip

# All known api names for compatibility with version of sip where
# `getapi` is not available ( < v4.9)
_API_NAMES = set(["QVariant", "QString", "QDate", "QDateTime",
                  "QTextStream", "QTime", "QUrl"])


def sip_getapi(name):
    """
    Get the api version for a name.
    """
    if sip.SIP_VERSION > 0x40900:
        return sip.getapi(name)
    elif name in _API_NAMES:
        return 1
    else:
        raise ValueError("unknown API {0!r}".format(name))


HAS_QVARIANT = sip_getapi("QVariant") == 1
HAS_QSTRING = sip_getapi("QString") == 1

if HAS_QVARIANT:
    from PyQt4.QtCore import QVariant

from PyQt4.QtCore import QSettings, QByteArray
from PyQt4.QtCore import PYQT_VERSION

QSETTINGS_HAS_TYPE = PYQT_VERSION >= 0x40803
"""QSettings.value has a `type` parameter"""


def toPyObject(variant):
    """Should not be needed with python 3. Please remove call to toPyObject"""
    raise NotImplementedError


if HAS_QVARIANT:
    toBitArray = methodcaller("toBitArray")
    toBool = methodcaller("toBool")
    toByteArray = methodcaller("toByteArray")
    toChar = methodcaller("toChar")
    toDate = methodcaller("")
    toPyObject = methodcaller("toPyObject")

    toFlaot = methodcaller("toFlaot")


def _check_error(xxx_todo_changeme):
    (val, status) = xxx_todo_changeme
    if not status:
        raise TypeError()
    else:
        return val


def qvariant_to_py(variant, py_type):
    """
    Convert a `QVariant` object to a python object of type `py_type`.
    """
    if py_type == bool:
        return variant.toBool()
    elif py_type == int:
        return _check_error(variant.toInt())
    elif py_type == str:
        return str(variant.toString())
    elif py_type == str:
        return str(variant.toString())
    elif py_type == QByteArray:
        return variant.toByteArray()

    else:
        raise TypeError("Unsuported type {0!s}".format(py_type))


if not QSETTINGS_HAS_TYPE:
    _QSettings = QSettings

    class QSettings(QSettings):
        """
        A subclass of QSettings with a simulated `type` parameter in
        value method.

        """
        # QSettings.value does not have `type` type before PyQt4 4.8.3
        # We dont't check if QVariant is exported, it is assumed on such old
        # installations the new api is not used.
        def value(self, key,
                  defaultValue=QVariant(),
                  type=None):
            """
            Returns the value for setting key. If the setting doesn't exist,
            returns defaultValue.

            """
            if not _QSettings.contains(self, key):
                return defaultValue

            value = _QSettings.value(self, key)

            if type is not None:
                value = qvariant_to_py(value, type)

            return value
