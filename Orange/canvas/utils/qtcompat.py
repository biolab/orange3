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

from PyQt4.QtCore import QSettings, QByteArray
from PyQt4.QtCore import PYQT_VERSION

QSETTINGS_HAS_TYPE = PYQT_VERSION >= 0x40803
"""QSettings.value has a `type` parameter"""


def toPyObject(variant):
    """Should not be needed with python 3. Please remove call to toPyObject"""
    raise NotImplementedError


def _check_error(xxx_todo_changeme):
    (val, status) = xxx_todo_changeme
    if not status:
        raise TypeError()
    else:
        return val
