from functools import reduce
from Orange.data.variable import TimeVariable


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
    try:
        return reduce(getattr, attr.split("."), obj)
    except AttributeError:
        if arg:
            return arg[0]
        if kwarg:
            return kwarg["default"]
        raise

def getHtmlCompatibleString(strVal):
    return strVal.replace("<=", "&#8804;").replace(">=","&#8805;").replace("<", "&#60;").replace(">","&#62;").replace("=\\=", "&#8800;")


# Utilities for inspecting/modifying widget geometry state as returned/set by
# QWidget's saveGeometry and restoreGeometry

import struct
from collections import namedtuple

#: Parsed geometry state (version 1; Qt < 5.4)
_geom_state_v1 = namedtuple(
    "geom_state",
    ["magic",   # int32_t magic constant
     "major_version", "minor_version",  # (uint16_t, uint16_t) format version
     #: 4x int32_t ( == frameGeometry().getCoords())
     "frame_left", "frame_top", "frame_right", "frame_bottom",
     #: 4x int32_t ( == geometry().getCoords() WHEN NOT in FullScreen)
     "normal_left", "normal_top", "normal_right", "normal_bottom",
     "screen",      # screen number
     "maximized",   # windowState() & Qt.WindowMaximized
     "full_screen"  # windowState() & Qt.WindowFullScreen
     ]
)


#: Parsed geometry state (version 2; Qt >= 5.4)
_geom_state_v2 = namedtuple(
    "geom_state",
    _geom_state_v1._fields +
    ("screen_width", )  # int32_t screen's width
)


def geom_state(magic, major_version, *args, **kwargs):
    if magic != geom_state.MAGIC:
        raise ValueError

    if major_version == 1:
        return geom_state.v1(magic, major_version, *args, **kwargs)
    elif major_version == 2:
        return geom_state.v2(magic, major_version, *args, **kwargs)
    else:
        raise ValueError

geom_state.MAGIC = 0x1D9D0CB
geom_state.v1 = _geom_state_v1
geom_state.v2 = _geom_state_v2


def geom_state_from_bytes(state):
    """
    Parse the QWidget.saveGeometry return value

    Parameters
    ----------
    state : bytes
        Saved widget geometry state as returned by :func:`QWidget.saveGeometry`

    Returns
    -------
    parsed : namedtuple
        The parsed geometry state as a named tuple
    """
    state = bytes(state)
    MAGIC = 0x1D9D0CB
    header_fmt = (
        "I"   # magic number
        "HH"  # minor.major format version
    )
    header_len = struct.calcsize(">" + header_fmt)
    magic, major, minor = struct.unpack(">" + header_fmt, state[:header_len])
    if magic != MAGIC:
        raise ValueError("Magic value does not match")
    if major not in {1, 2}:
        raise ValueError("Do not know how to handle version {}".format(major))

    payload_fmt = (
        "4i4i"  # 2x QRect's left, top, right, bottom
        "i"     # screen number
        "B"     # windowState() & Qt.WindowMaximized
        "B"     # windowState() & Qt.WindowFullScreen
    )

    if major == 2:
        payload_fmt += "i"  # screen's width

    payload_len = struct.calcsize(payload_fmt)
    geom = struct.unpack(
        ">" + payload_fmt, state[header_len: header_len + payload_len])
    return geom_state(magic, major, minor, *geom)


def geom_state_to_bytes(parsedstate):
    """
    Pack a parsed geometry state (geom_state) representation back to bytes

    Parameters
    ----------
    parsedstate : geom_state

    Returns
    -------
    bytes : bytes
    """
    fmt = ">IHH4i4iiBB"
    _, major = parsedstate[:2]
    if major >= 2:
        fmt += "i"

    return struct.pack(fmt, *parsedstate)

geom_state.from_bytes = geom_state_from_bytes
geom_state.v1.to_bytes = geom_state_to_bytes
geom_state.v2.to_bytes = geom_state_to_bytes


from PyQt4.QtCore import Qt, QRect, QMargins
from PyQt4.QtGui import QWidget


def geom_state_normal(state, widget_or_margins=None):
    """
    Strip the full screen flags from `state` returning it into normal mode

    Should be somewhat equivalent to .showNormal()

    Parameters
    ----------
    state : geom_state
        Parsed geometry state
    widget_or_margins : Union[QWidget, QMargins, Tuple[int, int, int, int]]
        Window frame margin hints

    Returns
    -------
    state : geom_state
    """
    def frame_margins(widget):
        frame = widget.frameGeometry()
        geom = widget.geometry()
        return (frame.left() - geom.left(),
                frame.top() - geom.top(),
                frame.right() - geom.right(),
                frame.bottom() - frame.bottom())

    if state.full_screen:
        # When a widget is in maximized/full screen mode the normal geometry
        # (as stored by saveGeometry) contains the widget's .geometry()
        # *before* it went to maximized/full screen (i.e. the geometry it
        # would/will get when restored back to normal mode). We use that to
        # recreate the frame geometry for the normal mode.
        frame = QRect()
        frame.setCoords(state.frame_left, state.frame_top,
                        state.frame_right, state.frame_bottom)
        normal = QRect()
        normal.setCoords(state.normal_left, state.normal_top,
                         state.normal_right, state.normal_bottom)

        if widget_or_margins is None:
            margins = 1, 20, 1, 1  # better guess?
        elif isinstance(widget_or_margins, QWidget):
            margins = frame_margins(widget_or_margins)
        elif isinstance(widget_or_margins, QMargins):
            margins = widget_or_margins
            margins = (margins.left(), margins.top(),
                       margins.right(), margins.bottom())
        else:
            margins = widget_or_margins

        left, top, right, bottom = margins
        frame = normal.adjusted(-left, -top, right, bottom)

        state = state._replace(
            frame_left=frame.left(), frame_top=frame.top(),
            frame_right=frame.right(), frame_bottom=frame.bottom(),
            full_screen=0, maximized=0
        )
    return state
