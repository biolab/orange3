import time
from os.path import dirname

from AnyQt.QtCore import Qt, QObject, pyqtSlot
from AnyQt.QtWidgets import QDialog
from AnyQt.QtTest import QTest

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.webview import WebviewWidget, HAVE_WEBKIT, wait

SOME_URL = WebviewWidget.toFileURL(dirname(__file__))


class WebviewWidgetTest(WidgetTest):
    def test_base(self):
        w = WebviewWidget()
        w.evalJS('document.write("foo");')
        SVG = '<svg xmlns:dc="...">asd</svg>'
        w.onloadJS('''document.write('{}');'''.format(SVG))
        w.setUrl(SOME_URL)

        svg = self.process_events(lambda: w.svg())
        self.assertEqual(svg, SVG)

        self.process_events(until=lambda: 'foo' in w.html())
        html = '<svg xmlns:dc="...">asd</svg>'
        self.assertEqual(
            w.html(),
            '<html><head></head><body>{}</body></html>'.format(
                # WebKit evaluates first document.write first, whereas
                # WebEngine evaluates onloadJS first
                'foo' + html if HAVE_WEBKIT else html + 'foo'
            ))

    def test_exposeObject(self):
        test = self
        OBJ = dict(a=[1, 2], b='c')
        done = False

        class Bridge(QObject):
            @pyqtSlot('QVariantMap')
            def check_object(self, obj):
                nonlocal test, done, OBJ
                done = True
                test.assertEqual(obj, OBJ)

        w = WebviewWidget(bridge=Bridge())
        w.setUrl(SOME_URL)
        w.exposeObject('obj', OBJ)
        w.evalJS('''pybridge.check_object(window.obj);''')
        self.process_events(lambda: done)

        self.assertRaises(ValueError, w.exposeObject, 'obj', QDialog())

    def test_escape_hides(self):
        # NOTE: This test doesn't work as it is supposed to.
        window = QDialog()
        w = WebviewWidget(window)
        window.show()
        w.setFocus(Qt.OtherFocusReason)
        self.assertFalse(window.isHidden())
        # This event is sent to the wrong widget. Should be sent to the
        # inner HTML view as focused, but no amount of clicking/ focusing
        # helped, neither did invoking JS handler directly. I'll live with it.
        QTest.keyClick(w, Qt.Key_Escape)
        self.assertTrue(window.isHidden())

    def test_runJavaScript(self):
        w = WebviewWidget()
        w.runJavaScript('2;')
        retvals = []
        w.runJavaScript('3;', lambda retval: retvals.append(retval))
        wait(until=lambda: retvals)
        self.assertEqual(retvals[0], 3)
