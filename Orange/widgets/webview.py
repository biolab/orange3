"""
This module holds our customized QWebView, an integration of HTML, CSS & JS
into Qt.
"""

from PyQt4.QtCore import Qt, QSize, QUrl
from PyQt4.QtGui import QSizePolicy, QWidget
from PyQt4.QtWebKit import QWebView


class WebView(QWebView):
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
    def __init__(self, parent=None, bridge=None, html='', *, debug=False, **kwargs):
        super().__init__(parent,
                         sizePolicy=QSizePolicy(QSizePolicy.Expanding,
                                                QSizePolicy.Expanding),
                         sizeHint=QSize(500, 400),
                         contextMenuPolicy=Qt.DefaultContextMenu,
                         **kwargs)

        if isinstance(parent, QWidget) and parent.layout() is not None:
            parent.layout().addWidget(self)

        self.bridge = bridge
        self.frame = None

        def _onload(_ok):
            if not _ok: return
            self.frame = self.page().mainFrame()
            self.frame.javaScriptWindowObjectCleared.connect(
                lambda: self.frame.addToJavaScriptWindowObject('pybridge', bridge))

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

        if html:
            self.setHtml(html)

    def setContent(self, data, mimetype, base_url=''):
        """Set the content `data` of type `mimetype` in the current webframe."""
        super().setContent(data, mimetype, QUrl(base_url))

    def dropEvent(self, event):
        """Prevent loading of drag-and-drop dropped file"""
        pass

    def evalJS(self, code):
        """Evaluate JavaScript code `code` in the current webframe and
        return the result of the last executed statement."""
        return self.frame.evaluateJavaScript(code)

    def setHtml(self, html, base_url=''):
        """Set the HTML content of the current webframe to `html`
        (an UTF-8 string)."""
        self.setContent(html.encode('utf-8'), 'text/html', base_url)

    def svg(self):
        """ Return SVG string of the first SVG element on the page, or
        raise ValueError if not any. """
        html = self.frame.toHtml()
        return html[html.index('<svg '):html.index('</svg>') + 6]
