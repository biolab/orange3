"""
Helper utilities

"""

import sys
import traceback

from contextlib import contextmanager
from functools import wraps

from AnyQt.QtWidgets import (
    QWidget, QMessageBox, QStyleOption, QStyle
)
from AnyQt.QtGui import (
    QGradient, QLinearGradient, QRadialGradient, QBrush, QPainter
)
from AnyQt.QtCore import QPointF, QUrl

import sip

QWIDGETSIZE_MAX = ((1 << 24) - 1)


# Serves as a decorator and context manager, hence a function-like name
class updates_disabled:  # pylint: disable=invalid-name
    """
    Disable QWidget updates (using QWidget.setUpdatesEnabled).

    `updates_disabled` can be used either as a context manager or a decorator
    for a method of a class derived from `QWidget`.

    When used as context manager, the argument is an instance of QWidget.

        with updates_disabled(self.plot_widget):
            do_something()

    When used as decorator, the argument is the name of the widget's
    attribute containing the child widget whose updates.

        @updated_disabled('plot_widget'):
        def method(self):
            do_something()

    In both cases, updates are enabled when the context or the function exits,
    and the widget's content is scheduled to be updated.
    """
    def __init__(self, widget):
        self.widget = widget

    def __enter__(self):
        self.old_state = self.widget.updatesEnabled()
        self.widget.setUpdatesEnabled(False)

    def __exit__(self, *exc):
        self.widget.setUpdatesEnabled(self.old_state)

    def __call__(self, method):
        @wraps(method)
        def func(widget, *args, **kwargs):
            with updates_disabled(getattr(widget, self.widget)):
                method(widget, *args, **kwargs)
        return func


@contextmanager
def signals_disabled(qobject):
    """Disables signals on an instance of QObject.
    """
    old_state = qobject.signalsBlocked()
    qobject.blockSignals(True)
    try:
        yield
    finally:
        qobject.blockSignals(old_state)


@contextmanager
def disabled(qobject):
    """Disables a disablable QObject instance.
    """
    if not (hasattr(qobject, "setEnabled") and hasattr(qobject, "isEnabled")):
        raise TypeError("%r does not have 'enabled' property" % qobject)

    old_state = qobject.isEnabled()
    qobject.setEnabled(False)
    try:
        yield
    finally:
        qobject.setEnabled(old_state)


def StyledWidget_paintEvent(self, event):
    """A default styled QWidget subclass  paintEvent function.
    """
    opt = QStyleOption()
    opt.initFrom(self)
    painter = QPainter(self)
    self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)


class StyledWidget(QWidget):
    """
    """
    paintEvent = StyledWidget_paintEvent


def is_transparency_supported():
    """Is window transparency supported by the current windowing system.

    """
    if sys.platform == "win32":
        return is_dwm_compositing_enabled()
    elif sys.platform == "cygwin":
        return False
    elif sys.platform == "darwin":
        if has_x11():
            return is_x11_compositing_enabled()
        else:
            # Quartz compositor
            return True
    elif sys.platform.startswith("linux"):
        # TODO: wayland??
        return is_x11_compositing_enabled()
    elif sys.platform.startswith("freebsd"):
        return is_x11_compositing_enabled()
    elif has_x11():
        return is_x11_compositing_enabled()
    else:
        return False


def has_x11():
    """
    Is Qt build against X11 server.
    """
    try:
        from AnyQt.QtX11Extras import QX11Info
        return True
    except ImportError:
        return False


def is_x11_compositing_enabled():
    """Is X11 compositing manager running.
    """
    try:
        from AnyQt.QtX11Extras import QX11Info
    except ImportError:
        return False
    if hasattr(QX11Info, "isCompositingManagerRunning"):
        return QX11Info.isCompositingManagerRunning()
    else:
        # not available on Qt5
        return False  # ?


def is_dwm_compositing_enabled():
    """Is Desktop Window Manager compositing (Aero) enabled.
    """
    import ctypes

    enabled = ctypes.c_bool()
    try:
        DwmIsCompositionEnabled = ctypes.windll.dwmapi.DwmIsCompositionEnabled
    except (AttributeError, WindowsError):
        # dwmapi or DwmIsCompositionEnabled is not present
        return False

    rval = DwmIsCompositionEnabled(ctypes.byref(enabled))

    return rval == 0 and enabled.value


def gradient_darker(grad, factor):
    """Return a copy of the QGradient darkened by factor.

    .. note:: Only QLinearGradeint and QRadialGradient are supported.

    """
    if type(grad) is QGradient:
        if grad.type() == QGradient.LinearGradient:
            grad = sip.cast(grad, QLinearGradient)
        elif grad.type() == QGradient.RadialGradient:
            grad = sip.cast(grad, QRadialGradient)

    if isinstance(grad, QLinearGradient):
        new_grad = QLinearGradient(grad.start(), grad.finalStop())
    elif isinstance(grad, QRadialGradient):
        new_grad = QRadialGradient(grad.center(), grad.radius(),
                                   grad.focalPoint())
    else:
        raise TypeError

    new_grad.setCoordinateMode(grad.coordinateMode())

    for pos, color in grad.stops():
        new_grad.setColorAt(pos, color.darker(factor))

    return new_grad


def brush_darker(brush, factor):
    """Return a copy of the brush darkened by factor.
    """
    grad = brush.gradient()
    if grad:
        return QBrush(gradient_darker(grad, factor))
    else:
        brush = QBrush(brush)
        brush.setColor(brush.color().darker(factor))
        return brush


def create_gradient(base_color, stop=QPointF(0, 0),
                    finalStop=QPointF(0, 1)):
    """
    Create a default linear gradient using `base_color` .

    """
    grad = QLinearGradient(stop, finalStop)
    grad.setStops([(0.0, base_color),
                   (0.5, base_color),
                   (0.8, base_color.darker(105)),
                   (1.0, base_color.darker(110)),
                   ])
    grad.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
    return grad


def create_css_gradient(base_color, stop=QPointF(0, 0),
                        finalStop=QPointF(0, 1)):
    """
    Create a Qt css linear gradient fragment based on the `base_color`.
    """
    gradient = create_gradient(base_color, stop, finalStop)
    return css_gradient(gradient)


def css_gradient(gradient):
    """
    Given an instance of a `QLinearGradient` return an equivalent qt css
    gradient fragment.

    """
    stop, finalStop = gradient.start(), gradient.finalStop()
    x1, y1, x2, y2 = stop.x(), stop.y(), finalStop.x(), finalStop.y()
    stops = gradient.stops()
    stops = "\n".join("    stop: {0:f} {1}".format(stop, color.name())
                      for stop, color in stops)
    return ("qlineargradient(\n"
            "    x1: {x1}, y1: {y1}, x2: {x1}, y2: {y2},\n"
            "{stops})").format(x1=x1, y1=y1, x2=x2, y2=y2, stops=stops)


def message_critical(text, title=None, informative_text=None, details=None,
                     buttons=None, default_button=None, exc_info=False,
                     parent=None):
    """Show a critical message.
    """
    if not text:
        text = "An unexpected error occurred."

    if title is None:
        title = "Error"

    return message(QMessageBox.Critical, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message_warning(text, title=None, informative_text=None, details=None,
                    buttons=None, default_button=None, exc_info=False,
                    parent=None):
    """Show a warning message.
    """
    if not text:
        import random
        text_candidates = ["Death could come at any moment.",
                           "Murphy lurks about. Remember to save frequently."
                           ]
        text = random.choice(text_candidates)

    if title is not None:
        title = "Warning"

    return message(QMessageBox.Warning, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message_information(text, title=None, informative_text=None, details=None,
                        buttons=None, default_button=None, exc_info=False,
                        parent=None):
    """Show an information message box.
    """
    if title is None:
        title = "Information"
    if not text:
        text = "I am not a number."

    return message(QMessageBox.Information, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message_question(text, title, informative_text=None, details=None,
                     buttons=None, default_button=None, exc_info=False,
                     parent=None):
    """Show an message box asking the user to select some
    predefined course of action (set by buttons argument).

    """
    return message(QMessageBox.Question, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message(icon, text, title=None, informative_text=None, details=None,
            buttons=None, default_button=None, exc_info=False, parent=None):
    """Show a message helper function.
    """
    if title is None:
        title = "Message"
    if not text:
        text = "I am neither a postman nor a doctor."

    if buttons is None:
        buttons = QMessageBox.Ok

    if details is None and exc_info:
        details = traceback.format_exc(limit=20)

    mbox = QMessageBox(icon, title, text, buttons, parent)

    if informative_text:
        mbox.setInformativeText(informative_text)

    if details:
        mbox.setDetailedText(details)

    if default_button is not None:
        mbox.setDefaultButton(default_button)

    return mbox.exec_()


def OSX_NSURL_toLocalFile(url):
    """Return OS X NSURL file reference as local file path or '' if not NSURL"""
    if isinstance(url, QUrl):
        url = url.toString()
    if not url.startswith('file:///.file/id='):
        return ''
    from subprocess import Popen, PIPE, DEVNULL
    cmd = ['osascript', '-e', 'get POSIX path of POSIX file "{}"'.format(url)]
    with Popen(cmd, stdout=PIPE, stderr=DEVNULL) as p:
        return p.stdout.read().strip().decode()
