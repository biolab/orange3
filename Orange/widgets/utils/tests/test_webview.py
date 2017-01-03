from os.path import dirname

from AnyQt.QtCore import Qt, QObject, pyqtSlot
from AnyQt.QtWidgets import QDialog, qApp
from AnyQt.QtTest import QTest

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.webview import WebviewWidget

SOME_URL = WebviewWidget.toFileURL(dirname(__file__))


class WebviewWidgetTest(WidgetTest):
    def test_base(self):
        w = WebviewWidget()
        w.evalJS('document.write("foo");')
        SVG = '<svg xmlns:dc="...">asd</svg>'
        w.onloadJS('''document.write('{}');'''.format(SVG))
        w.setUrl(SOME_URL)

        svg = None
        while svg is None:
            try:
                svg = w.svg()
                break
            except ValueError:
                qApp.processEvents()
        self.assertEqual(svg, SVG)
        self.assertEqual(
            w.html(), '<html><head></head><body>foo<svg xmlns:dc="...">asd</svg></body></html>')

    def test_exposeObject(self):
        test = self
        OBJ = dict(a=[1, 2], b='c')

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

        done = False
        while not done:
            qApp.processEvents()

        self.assertRaises(ValueError, w.exposeObject, 'obj', QDialog())

    def test_escape_hides(self):
        window = QDialog()
        w = WebviewWidget(window)
        window.show()
        w.setFocus(Qt.OtherFocusReason)
        self.assertFalse(window.isHidden())
        QTest.keyClick(w, Qt.Key_Escape)
        self.assertTrue(window.isHidden())
