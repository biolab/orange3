import unittest

from PyQt4.QtGui import QApplication

app = None


class GuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global app
        if app is None:
            app = QApplication([])

