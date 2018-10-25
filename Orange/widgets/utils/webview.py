"""
This module holds our customized WebView that integrates HTML, CSS & JS
into Qt. WebviewWidget provides a somewhat uniform interface (_WebViewBase)
around either WebEngineView (extends QWebEngineView) or WebKitView
(extends QWebView), as available.
"""
import os
import time
import threading
import warnings
from collections.abc import Iterable, Mapping, Set, Sequence
from itertools import count
from numbers import Integral, Real
from os.path import abspath, dirname, join
from random import random
from urllib.parse import urljoin
from urllib.request import pathname2url

import numpy as np
import sip

from Orange.util import inherit_docstrings, OrangeDeprecationWarning

from AnyQt.QtCore import Qt, QObject, QFile, QTimer, QUrl, QSize, QEventLoop, \
    pyqtProperty, pyqtSlot, pyqtSignal
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QSizePolicy, QWidget, qApp

try:
    from AnyQt.QtWebKitWidgets import QWebView
    HAVE_WEBKIT = True
except ImportError:
    HAVE_WEBKIT = False

try:
    from AnyQt.QtWebEngineWidgets import QWebEngineView, QWebEngineScript
    from AnyQt.QtWebChannel import QWebChannel
    HAVE_WEBENGINE = True
except ImportError:
    HAVE_WEBENGINE = False


_WEBVIEW_HELPERS = join(dirname(__file__), '_webview', 'helpers.js')
_WEBENGINE_INIT_WEBCHANNEL = join(dirname(__file__), '_webview', 'init-webengine-webchannel.js')

_ORANGE_DEBUG = os.environ.get('ORANGE_DEBUG')


class _QWidgetJavaScriptWrapper(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.__parent = parent

    @pyqtSlot()
    def load_really_finished(self):
        self.__parent._load_really_finished()

    @pyqtSlot()
    def hideWindow(self):
        w = self.__parent
        while isinstance(w, QWidget):
            if w.windowFlags() & (Qt.Window | Qt.Dialog):
                return w.hide()
            w = w.parent() if callable(w.parent) else w.parent


if HAVE_WEBENGINE:
    class WebEngineView(QWebEngineView):
        """
        A QWebEngineView initialized to support communication with JS code.

        Parameters
        ----------
        parent : QWidget
            Parent widget object.
        bridge : QObject
            The "bridge" object exposed as a global object ``pybridge`` in
            JavaScript. Any methods desired to be accessible from JS need
            to be decorated with ``@QtCore.pyqtSlot(<*args>, result=<type>)``
            decorator.
            Note: do not use QWidget instances as a bridge, use a minimal
            QObject subclass implementing the required interface.
        """

        # Prefix added to objects exposed via WebviewWidget.exposeObject()
        # This caters to this class' subclass
        _EXPOSED_OBJ_PREFIX = '__ORANGE_'

        def __init__(self, parent=None, bridge=None, *, debug=False, **kwargs):
            debug = debug or _ORANGE_DEBUG
            if debug:
                port = os.environ.setdefault('QTWEBENGINE_REMOTE_DEBUGGING', '12088')
                warnings.warn(
                    'To debug QWebEngineView, set environment variable '
                    'QTWEBENGINE_REMOTE_DEBUGGING={port} and then visit '
                    'http://127.0.0.1:{port}/ in a Chromium-based browser. '
                    'See https://doc.qt.io/qt-5/qtwebengine-debugging.html '
                    'This has also been done for you.'.format(port=port))
            super().__init__(parent,
                             sizeHint=QSize(500, 400),
                             sizePolicy=QSizePolicy(QSizePolicy.Expanding,
                                                    QSizePolicy.Expanding),
                             **kwargs)
            self.bridge = bridge
            self.debug = debug
            with open(_WEBVIEW_HELPERS, encoding="utf-8") as f:
                self._onloadJS(f.read(),
                               name='webview_helpers',
                               injection_point=QWebEngineScript.DocumentCreation)

            qtwebchannel_js = QFile("://qtwebchannel/qwebchannel.js")
            if qtwebchannel_js.open(QFile.ReadOnly):
                source = bytes(qtwebchannel_js.readAll()).decode("utf-8")
                with open(_WEBENGINE_INIT_WEBCHANNEL, encoding="utf-8") as f:
                    init_webchannel_src = f.read()
                self._onloadJS(source + init_webchannel_src %
                               dict(exposeObject_prefix=self._EXPOSED_OBJ_PREFIX),
                               name='webchannel_init',
                               injection_point=QWebEngineScript.DocumentCreation)
            else:
                warnings.warn(
                    "://qtwebchannel/qwebchannel.js is not readable.",
                    RuntimeWarning)

            self._onloadJS(';window.__load_finished = true;',
                           name='load_finished',
                           injection_point=QWebEngineScript.DocumentReady)

            channel = QWebChannel(self)
            if bridge is not None:
                if isinstance(bridge, QWidget):
                    warnings.warn(
                        "Don't expose QWidgets in WebView. Construct minimal "
                        "QObjects instead.", OrangeDeprecationWarning,
                        stacklevel=2)
                channel.registerObject("pybridge", bridge)

            channel.registerObject('__bridge', _QWidgetJavaScriptWrapper(self))

            self.page().setWebChannel(channel)

        def _onloadJS(self, code, name='', injection_point=QWebEngineScript.DocumentReady):
            script = QWebEngineScript()
            script.setName(name or ('script_' + str(random())[2:]))
            script.setSourceCode(code)
            script.setInjectionPoint(injection_point)
            script.setWorldId(script.MainWorld)
            script.setRunsOnSubFrames(False)
            self.page().scripts().insert(script)
            self.loadStarted.connect(
                lambda: self.page().scripts().insert(script))

        def runJavaScript(self, javascript, resultCallback=None):
            """
            Parameters
            ----------
            javascript : str
                javascript code.
            resultCallback : Optional[(object) -> None]
                When the script has been executed the `resultCallback` will
                be called with the result of the last executed statement.
            """
            if resultCallback is not None:
                self.page().runJavaScript(javascript, resultCallback)
            else:
                self.page().runJavaScript(javascript)


if HAVE_WEBKIT:
    class WebKitView(QWebView):
        """
        Construct a new QWebView widget that has no history and
        supports loading from local URLs.

        Parameters
        ----------
        parent: QWidget
            The parent widget.
        bridge: QObject
            The QObject to use as a parent. This object is also exposed
            as ``window.pybridge`` in JavaScript.
        html: str
            The HTML to load the view with.
        debug: bool
            Whether to enable inspector tools on right click.
        **kwargs:
            Passed to QWebView.
        """
        def __init__(self, parent=None, bridge=None, *, debug=False, **kwargs):
            super().__init__(parent,
                             sizeHint=QSize(500, 400),
                             sizePolicy=QSizePolicy(QSizePolicy.Expanding,
                                                    QSizePolicy.Expanding),
                             **kwargs)

            if isinstance(parent, QWidget) and parent.layout() is not None:
                parent.layout().addWidget(self)  # TODO REMOVE

            self.bridge = bridge
            self.frame = None
            debug = debug or _ORANGE_DEBUG
            self.debug = debug

            if isinstance(bridge, QWidget):
                warnings.warn(
                    "Don't expose QWidgets in WebView. Construct minimal "
                    "QObjects instead.", OrangeDeprecationWarning,
                    stacklevel=2)

            def _onload(_ok):
                if _ok:
                    self.frame = self.page().mainFrame()
                    self.frame.javaScriptWindowObjectCleared.connect(
                        lambda: self.frame.addToJavaScriptWindowObject('pybridge', bridge))
                    with open(_WEBVIEW_HELPERS, encoding="utf-8") as f:
                        self.frame.evaluateJavaScript(f.read())

            self.loadFinished.connect(_onload)
            _onload(True)

            history = self.history()
            history.setMaximumItemCount(0)
            settings = self.settings()
            settings.setMaximumPagesInCache(0)
            settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, False)

            if debug:
                settings.setAttribute(settings.LocalStorageEnabled, True)
                settings.setAttribute(settings.DeveloperExtrasEnabled, True)
                settings.setObjectCacheCapacities(4e6, 4e6, 4e6)
                settings.enablePersistentStorage()

        def runJavaScript(self, javascript, resultCallback=None):
            result = self.page().mainFrame().evaluateJavaScript(javascript)
            if resultCallback is not None:
                # Emulate the QtWebEngine's interface and return the result
                # in a event queue invoked callback
                QTimer.singleShot(0, lambda: resultCallback(result))


def _to_primitive_types(d):
    # pylint: disable=too-many-return-statements
    if isinstance(d, QWidget):
        raise ValueError("Don't expose QWidgets in WebView. Construct minimal "
                         "QObjects instead.")
    if isinstance(d, Integral):
        return int(d)
    if isinstance(d, Real):
        return float(d)
    if isinstance(d, (bool, np.bool_)):
        return bool(d)
    if isinstance(d, (str, QObject)):
        return d
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, Mapping):
        return {k: _to_primitive_types(d[k]) for k in d}
    if isinstance(d, Set):
        return {k: 1 for k in d}
    if isinstance(d, (Sequence, Iterable)):
        return [_to_primitive_types(i) for i in d]
    if d is None:
        return None
    if isinstance(d, QColor):
        return d.name()
    raise TypeError(
        'object must consist of primitive types '
        '(allowed: int, float, str, bool, list, '
        'dict, set, numpy.ndarray, ...). Type is: ' + d.__class__)


class _WebViewBase:
    def _evalJS(self, code):
        """Evaluate JavaScript code and return the result of the last statement."""
        raise NotImplementedError

    def onloadJS(self, code):
        """Run JS on document load."""
        raise NotImplementedError

    def html(self):
        """Return HTML contents of the top frame.

        Warnings
        --------
        In the case of Qt WebEngine implementation, this function calls:

            QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        until the page's HTML contents is made available (through IPC).
        """
        raise NotImplementedError

    def exposeObject(self, name, obj):
        """Expose the object `obj` as ``window.<name>`` in JavaScript.

        If the object contains any string values that start and end with
        literal ``/**/``, those are evaluated as JS expressions the result
        value replaces the string in the object.

        The exposure, as defined here, represents a snapshot of object at
        the time of execution. Any future changes on the original Python
        object are not visible in its JavaScript counterpart.

        Parameters
        ----------
        name: str
            The global name the object is exposed as.
        obj: object
            The object to expose. Must contain only primitive types, such as:
            int, float, str, bool, list, dict, set, numpy.ndarray ...
        """
        raise NotImplementedError

    def __init__(self):
        self.__is_init = False
        self.__js_queue = []

    @pyqtSlot()
    def _load_really_finished(self):
        """Call this from JS when the document is ready."""
        self.__is_init = True

    def dropEvent(self, event):
        """Prevent loading of drag-and-drop dropped file"""
        pass

    def evalJS(self, code):
        """
        Evaluate JavaScript code synchronously (or sequentially, at least).

        Parameters
        ----------
        code : str
            The JavaScript code to evaluate in main page frame. The scope is
            not assured. Assign properties to window if you want to make them
            available elsewhere.

        Warnings
        --------
        In the case of Qt WebEngine implementation, this function calls:

            QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        until the page is fully loaded, all the objects exposed via
        exposeObject() method are indeed exposed in JS, and the code `code`
        has finished evaluating.
        """
        def _later():
            if not self.__is_init and self.__js_queue:
                return QTimer.singleShot(1, _later)
            if self.__js_queue:
                # '/n' is required when the last line is a comment
                code = '\n;'.join(self.__js_queue)
                self.__js_queue.clear()
                self._evalJS(code)

        # WebView returns the result of the last evaluated expression.
        # This result may be too complex an object to safely receive on this
        # end, so instead, just make it return 0.
        code += ';0;'
        self.__js_queue.append(code)
        QTimer.singleShot(1, _later)

    def svg(self):
        """ Return SVG string of the first SVG element on the page, or
        raise ValueError if not any. """
        html = self.html()
        return html[html.index('<svg '):html.index('</svg>') + 6]

    def setHtml(self, html, base_url=''):
        """Set the HTML content of the current webframe to `html`
        (an UTF-8 string)."""
        super().setHtml(html, QUrl(base_url))

    @staticmethod
    def toFileURL(local_path):
        """Return local_path as file:// URL"""
        return urljoin('file:', pathname2url(abspath(local_path)))

    def setUrl(self, url):
        """Point the current frame to URL url."""
        super().setUrl(QUrl(url))

    def contextMenuEvent(self, event):
        """ Also disable context menu unless debug."""
        if self.debug:
            super().contextMenuEvent(event)


def wait(until: callable, timeout=5000):
    """Process events until condition is satisfied

    Parameters
    ----------
    until: callable
        Returns True when condition is satisfied.
    timeout: int
        Milliseconds to wait until TimeoutError is raised.
    """
    started = time.clock()
    while not until():
        qApp.processEvents(QEventLoop.ExcludeUserInputEvents)
        if (time.clock() - started) * 1000 > timeout:
            raise TimeoutError()


if HAVE_WEBKIT:

    class _JSObject(QObject):
        """ This class hopefully prevent options data from being
        marshalled into a string-like dumb (JSON) object when
        passed into JavaScript. Or at least relies on Qt to do it as
        optimally as it knows to."""

        def __init__(self, parent, name, obj):
            super().__init__(parent)
            self._obj = dict(obj=obj)

        @pyqtProperty('QVariantMap')
        def pop_object(self):
            return self._obj

    @inherit_docstrings
    class WebviewWidget(_WebViewBase, WebKitView):
        def __init__(self, parent=None, bridge=None, *, debug=False, **kwargs):
            WebKitView.__init__(self, parent, bridge, debug=debug, **kwargs)
            _WebViewBase.__init__(self)

            def load_finished():
                if not sip.isdeleted(self):
                    self.frame.addToJavaScriptWindowObject(
                        '__bridge', _QWidgetJavaScriptWrapper(self))
                    self._evalJS('setTimeout(function(){'
                                 '__bridge.load_really_finished(); }, 100);')

            self.loadFinished.connect(load_finished)

        @pyqtSlot()
        def _load_really_finished(self):
            # _WebViewBase's (super) method not visible in JS for some reason
            super()._load_really_finished()

        def _evalJS(self, code):
            return self.frame.evaluateJavaScript(code)

        def onloadJS(self, code):
            self.frame.loadFinished.connect(
                lambda: WebviewWidget.evalJS(self, code))

        def html(self):
            return self.frame.toHtml()

        def exposeObject(self, name, obj):
            obj = _to_primitive_types(obj)
            jsobj = _JSObject(self, name, obj)
            self.frame.addToJavaScriptWindowObject('__js_object_' + name, jsobj)
            WebviewWidget.evalJS(self, '''
                window.{0} = window.__js_object_{0}.pop_object.obj;
                fixupPythonObject({0}); 0;
            '''.format(name))


elif HAVE_WEBENGINE:
    class IdStore:
        """Generates and stores unique ids.

        Used in WebviewWidget._evalJS below to match scheduled js executions
        and returned results. WebEngine operations are async, so locking is
        used to guard against problems that could occur if multiple executions
        ended at exactly the same time.
        """

        def __init__(self):
            self.id = 0
            self.lock = threading.Lock()
            self.ids = dict()

        def create(self):
            with self.lock:
                self.id += 1
                return self.id

        def store(self, id, value):
            with self.lock:
                self.ids[id] = value

        def __contains__(self, id):
            return id in self.ids

        def pop(self, id):
            with self.lock:
                return self.ids.pop(id, None)


    class _JSObjectChannel(QObject):
        """ This class hopefully prevent options data from being
        marshalled into a string-like dumb (JSON) object when
        passed into JavaScript. Or at least relies on Qt to do it as
        optimally as it knows to."""

        # JS webchannel listens to this signal
        objectChanged = pyqtSignal('QVariantMap')

        def __init__(self, parent):
            super().__init__(parent)
            self._obj = None
            self._id_gen = count()
            self._objects = {}

        def send_object(self, name, obj):
            if isinstance(obj, QObject):
                raise ValueError(
                    "QWebChannel doesn't transmit QObject instances. If you "
                    "need a QObject available in JavaScript, pass it as a "
                    "bridge in WebviewWidget constructor.")
            id = next(self._id_gen)
            value = self._objects[id] = dict(id=id, name=name, obj=obj)
            # Wait till JS is connected to receive objects
            wait(until=lambda: self.receivers(self.objectChanged))
            self.objectChanged.emit(value)

        @pyqtSlot(int)
        def mark_exposed(self, id):
            del self._objects[id]

        def is_all_exposed(self):
            return len(self._objects) == 0


    _NOTSET = object()


    @inherit_docstrings
    class WebviewWidget(_WebViewBase, WebEngineView):
        _html = _NOTSET

        def __init__(self, parent=None, bridge=None, *, debug=False, **kwargs):
            WebEngineView.__init__(self, parent, bridge, debug=debug, **kwargs)
            _WebViewBase.__init__(self)

            # Tracks objects exposed in JS via exposeObject(). JS notifies once
            # the object has indeed been exposed (i.e. the new object is available
            # is JS) because the whole thing is async.
            # This is needed to stall any evalJS() calls which may expect
            # the objects being available (but could otherwise be executed before
            # the objects are exposed in JS).
            self._jsobject_channel = jsobj = _JSObjectChannel(self)
            self.page().webChannel().registerObject(
                '__js_object_channel', jsobj)
            self._results = IdStore()

        def _evalJS(self, code):
            wait(until=self._jsobject_channel.is_all_exposed)
            if sip.isdeleted(self):
                return None
            result = self._results.create()
            self.runJavaScript(code, lambda x: self._results.store(result, x))
            wait(until=lambda: result in self._results)
            return self._results.pop(result)

        def onloadJS(self, code):
            self._onloadJS(code, injection_point=QWebEngineScript.Deferred)

        def html(self):
            self.page().toHtml(lambda html: setattr(self, '_html', html))
            wait(until=lambda: self._html is not _NOTSET or sip.isdeleted(self))
            html, self._html = self._html, _NOTSET
            return html

        def exposeObject(self, name, obj):
            obj = _to_primitive_types(obj)
            self._jsobject_channel.send_object(name, obj)

        def setHtml(self, html, base_url=''):
            # TODO: remove once anaconda will provide PyQt without this bug.
            #
            # At least on some installations of PyQt 5.6.0 with anaconda
            # WebViewWidget grabs focus on setHTML which can be quite annoying.
            # For example, if you have a line edit as filter and show results
            # in WebWiew, then WebView grabs focus after every typed character.
            #
            # http://stackoverflow.com/questions/36609489
            # https://bugreports.qt.io/browse/QTBUG-52999
            initial_state = self.isEnabled()
            self.setEnabled(False)
            super().setHtml(html, base_url)
            self.setEnabled(initial_state)
