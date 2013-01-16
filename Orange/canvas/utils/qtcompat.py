"""
PyQt4 compatibility utility functions

"""
import sip

# All known api names for compatibility with version of sip where
# `getapi` is not available ( < v4.9)
_API_NAMES = set(["QVariant", "QString", "QDate", "QDateTime",
                  "QTextStream", "QTime", "QUrl"])


def sip_getapi(name):
    """
    Get the api version for a name.
    """
    if sip.SIP_VERSION < 0x40900:
        return sip.getapi(name)
    elif name in _API_NAMES:
        return 1
    else:
        raise ValueError("unknown API {0!r}".format(name))


def toPyObject(obj):
    """
    Return `obj` as a python object if it is wrapped in a `QVariant`
    instance (using `obj.toPyObject()`). In case the sip API version for
    QVariant does not export it just return the object unchanged.

    """
    if sip_getapi("QVariant") > 1:
        return obj
    else:
        try:
            return obj.toPyObject()
        except AttributeError:
            return obj
