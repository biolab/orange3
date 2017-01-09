import os
import pip
import sys
import time
import logging
import platform
import traceback
import uuid
from html import escape
from threading import Thread
from pprint import pformat

from tempfile import mkstemp
from collections import OrderedDict
from urllib.parse import urljoin, urlencode
from urllib.request import pathname2url, urlopen
from unittest.mock import patch

from AnyQt.QtCore import pyqtSlot, QSettings, Qt
from AnyQt.QtGui import QDesktopServices, QFont
from AnyQt.QtWidgets import (
    QApplication, QCheckBox, QDialog, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QStyle, QTextBrowser,
    QVBoxLayout, QWidget
)

try:
    from Orange.widgets.widget import OWWidget
    from Orange.version import full_version as VERSION_STR
except ImportError:
    # OWWidget (etc.) is not available because this is not Orange
    class OWWidget: pass
    VERSION_STR = '???'


REPORT_POST_URL = 'http://orange.biolab.si/error_report/v1/'

log = logging.getLogger()


class ErrorReporting(QDialog):
    _cache = set()  # For errors already handled during one session

    class DataField:
        EXCEPTION = 'Exception'
        MODULE = 'Module'
        WIDGET_NAME = 'Widget Name'
        WIDGET_MODULE = 'Widget Module'
        VERSION = 'Version'
        ENVIRONMENT = 'Environment'
        INSTALLED_PACKAGES = 'Installed Packages'
        MACHINE_ID = 'Machine ID'
        WIDGET_SCHEME = 'Widget Scheme'
        STACK_TRACE = 'Stack Trace'
        LOCALS = 'Local Variables'

    def __init__(self, data):
        icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)
        F = self.DataField

        def _finished(*, key=(data.get(F.MODULE),
                              data.get(F.WIDGET_MODULE)),
                      filename=data.get(F.WIDGET_SCHEME)):
            self._cache.add(key)
            try: os.remove(filename)
            except Exception: pass

        super().__init__(None, Qt.Window, modal=True,
                         sizeGripEnabled=True, windowIcon=icon,
                         windowTitle='Unexpected Error',
                         finished=_finished)
        self._data = data

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        labels = QWidget(self)
        labels_layout = QHBoxLayout(self)
        labels.setLayout(labels_layout)
        labels_layout.addWidget(QLabel(pixmap=icon.pixmap(50, 50)))
        labels_layout.addWidget(QLabel(
            'The program encountered an unexpected error. Please<br>'
            'report it anonymously to the developers.<br><br>'
            'The following data will be reported:'))
        labels_layout.addStretch(1)
        layout.addWidget(labels)
        font = QFont('Monospace', 10)
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        textbrowser = QTextBrowser(self,
                                   font=font,
                                   openLinks=False,
                                   lineWrapMode=QTextBrowser.NoWrap,
                                   anchorClicked=QDesktopServices.openUrl)
        layout.addWidget(textbrowser)

        def _reload_text():
            add_scheme = cb.isChecked()
            settings.setValue('error-reporting/add-scheme', add_scheme)
            lines = ['<table>']
            for k, v in data.items():
                if k.startswith('_'):
                    continue
                _v, v = v, escape(v)
                if k == F.WIDGET_SCHEME:
                    if not add_scheme:
                        continue
                    v = '<a href="{}">{}</a>'.format(urljoin('file:', pathname2url(_v)), v)
                if k in (F.STACK_TRACE, F.LOCALS):
                    v = v.replace('\n', '<br>').replace(' ', '&nbsp;')
                lines.append('<tr><th align="left">{}:</th><td>{}</td></tr>'.format(k, v))
            lines.append('</table>')
            textbrowser.setHtml(''.join(lines))

        settings = QSettings()
        cb = QCheckBox(
            'Include workflow (data will NOT be transmitted)', self,
            checked=settings.value('error-reporting/add-scheme', True, type=bool))
        cb.stateChanged.connect(_reload_text)
        _reload_text()

        layout.addWidget(cb)
        buttons = QWidget(self)
        buttons_layout = QHBoxLayout(self)
        buttons.setLayout(buttons_layout)
        buttons_layout.addWidget(QPushButton('Send Report (Thanks!)', default=True, clicked=self.accept))
        buttons_layout.addWidget(QPushButton("Don't Send", default=False, clicked=self.reject))
        layout.addWidget(buttons)

    def accept(self):
        super().accept()
        F = self.DataField
        data = self._data.copy()

        if QSettings().value('error-reporting/add-scheme', True, type=bool):
            data[F.WIDGET_SCHEME] = data['_' + F.WIDGET_SCHEME]
        del data['_' + F.WIDGET_SCHEME]

        def _post_report(data):
            MAX_RETRIES = 2
            for _retry in range(MAX_RETRIES):
                try:
                    urlopen(REPORT_POST_URL,
                            timeout=10,
                            data=urlencode(data).encode('utf8'))
                except Exception as e:
                    if _retry == MAX_RETRIES - 1:
                        e.__context__ = None
                        log.exception('Error reporting failed', exc_info=e)
                    time.sleep(10)
                    continue
                break

        Thread(target=_post_report, args=(data,)).start()

    @classmethod
    @patch('sys.excepthook', sys.__excepthook__)  # Prevent recursion
    @pyqtSlot(object)
    def handle_exception(cls, exc):
        (etype, evalue, tb), canvas = exc
        exception = traceback.format_exception_only(etype, evalue)[-1].strip()
        stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))

        def _find_last_frame(tb):
            if not tb:
                return None
            while tb.tb_next:
                tb = tb.tb_next
            return tb

        err_locals, err_module, frame = None, None, _find_last_frame(tb)
        if frame:
            err_module = '{}:{}'.format(
                frame.tb_frame.f_globals.get('__name__', frame.tb_frame.f_code.co_filename),
                frame.tb_lineno)
            err_locals = pformat(OrderedDict(sorted(frame.tb_frame.f_locals.items())))

        def _find_widget_frame(tb):
            while tb:
                if isinstance(tb.tb_frame.f_locals.get('self'), OWWidget):
                    return tb
                tb = tb.tb_next

        widget_module, widget, frame = None, None, _find_widget_frame(tb)
        if frame:
            widget = frame.tb_frame.f_locals['self'].__class__
            widget_module = '{}:{}'.format(widget.__module__, frame.tb_lineno)

        packages = ', '.join(sorted("%s==%s" % (i.project_name, i.version)
                                    for i in pip.get_installed_distributions()))

        # If this exact error was already reported in this session,
        # just warn about it
        if (err_module, widget_module) in cls._cache:
            QMessageBox(QMessageBox.Warning, 'Error Encountered',
                        'Error encountered{}:<br><br><tt>{}</tt>'.format(
                            ' in widget <b>{}</b>'.format(widget.name) if widget else '',
                            stacktrace.replace('\n', '<br>').replace(' ', '&nbsp;')),
                        QMessageBox.Ignore).exec()
            return

        F = cls.DataField
        data = OrderedDict()
        data[F.EXCEPTION] = exception
        data[F.MODULE] = err_module
        if widget:
            data[F.WIDGET_NAME] = widget.name
            data[F.WIDGET_MODULE] = widget_module
        if canvas:
            fd, filename = mkstemp(prefix='ows-', suffix='.ows.xml')
            os.close(fd)
            # Prevent excepthook printing the same exception when
            # canvas tries to instantiate the broken widget again
            with patch('sys.excepthook', lambda *_: None), \
                    open(filename, "wb") as f:
                scheme = canvas.current_document().scheme()
                try:
                    scheme.save_to(f, pretty=True, pickle_fallback=True)
                except Exception:
                    pass
            data[F.WIDGET_SCHEME] = filename
            with open(filename) as f:
                data['_' + F.WIDGET_SCHEME] = f.read()
        data[F.VERSION] = VERSION_STR
        data[F.ENVIRONMENT] = 'Python {} on {} {} {} {}'.format(
            platform.python_version(), platform.system(), platform.release(),
            platform.version(), platform.machine())
        data[F.INSTALLED_PACKAGES] = packages
        data[F.MACHINE_ID] = str(uuid.getnode())
        data[F.STACK_TRACE] = stacktrace
        if err_locals:
            data[F.LOCALS] = err_locals

        cls(data=data).exec()


def handle_exception(exc):
    return ErrorReporting.handle_exception(exc)
