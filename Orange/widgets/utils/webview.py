import textwrap
import warnings

from AnyQt.QtCore import Qt, QObject, QFile, QTimer, QUrl, QT_VERSION

#: Is QtWebKitWidgets available
HAVE_WEBKIT = False
try:
    from AnyQt import QtWebKitWidgets
    HAVE_WEBKIT = True
except ImportError:
    pass

#: Is QtWebEngineWidgets (with injectable QWebEngineScript) available
HAVE_WEBENGINE = False
if QT_VERSION >= 0x50500:
    try:
        from PyQt5 import QtWebEngineWidgets, QtWebChannel
        HAVE_WEBENGINE = True
    except ImportError:
        pass


if HAVE_WEBENGINE:
    class WebEngineView(QtWebEngineWidgets.QWebEngineView):
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
        def __init__(self, parent=None, bridge=None, **kwargs):
            super().__init__(parent, **kwargs)
            self._bridge = bridge
            channel = QtWebChannel.QWebChannel(self)
            if self._bridge is not None:
                channel.registerObject("pybridge", self._bridge)
            self.page().setWebChannel(channel)

            qtwebchannel_js = QFile("://qtwebchannel/qwebchannel.js")
            if qtwebchannel_js.open(QFile.ReadOnly):
                source = bytes(qtwebchannel_js.readAll()).decode("utf-8")
                script = QtWebEngineWidgets.QWebEngineScript()
                script.setName("qwebchannel")
                script.setSourceCode(source)
                script.setInjectionPoint(script.DocumentCreation)
                script.setWorldId(script.MainWorld)
                script.setRunsOnSubFrames(False)
                self.page().scripts().insert(script)

                script = QtWebEngineWidgets.QWebEngineScript()
                script.setName("pybridgeinit")
                script.setSourceCode(textwrap.dedent("""
                var pybridge = null;
                new QWebChannel(qt.webChannelTransport,
                    function(channel) {
                        pybridge = channel.objects.pybridge;
                    }
                );
                """))
                script.setInjectionPoint(script.DocumentReady)
                script.setWorldId(script.MainWorld)
                script.setRunsOnSubFrames(False)
                if self._bridge is not None:
                    self.page().scripts().insert(script)
            else:
                warnings.warn(
                    "://qtwebchannel/qwebchannel.js is not readable.",
                    RuntimeWarning)

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
    class WebKitView(QtWebKitWidgets.QWebView):
        """
        'Similar' to WebEngineView but using QWebView.
        """
        def __init__(self, parent=None, bridge=None, **kwargs):
            super().__init__(parent, **kwargs)
            self._bridge = bridge
            settings = self.settings()
            settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)

        def setContent(self, data, mimetype, url=QUrl()):
            super().setContent(data, mimetype, QUrl(url))
            if self._bridge is not None:
                # TODO: Use QWebFrame.javaScriptWindowObjectCleared
                page = self.page()
                page.mainFrame().addToJavaScriptWindowObject(
                    'pybridge', self._bridge)

        def setHtml(self, html, url=QUrl()):
            self.setContent(html.encode('utf-8'), 'text/html', url)

        def runJavaScript(self, javascript, resultCallback=None):
            result = self.page().mainFrame().evaluateJavaScript(javascript)
            if resultCallback is not None:
                # Emulate the QtWebEngine's interface and return the result
                # in a event queue invoked callback
                QTimer.singleShot(0, lambda: resultCallback(result))
