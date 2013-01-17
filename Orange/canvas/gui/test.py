"""
Basic Qt testing framework
==========================
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import gc

from PyQt4.QtGui import QApplication
from PyQt4.QtCore import QCoreApplication, QTimer


class QAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = QApplication([])
        QTimer.singleShot(20000, self.app.exit)

    def tearDown(self):
        if hasattr(self, "scene"):
            self.scene.clear()
            self.scene.deleteLater()
            self.app.processEvents()
            del self.scene
        self.app.processEvents()
        del self.app
        gc.collect()

    def singleShot(self, *args):
        QTimer.singleShot(*args)


class QCoreAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication([])
        QTimer.singleShot(20000, self.app.exit)

    def tearDown(self):
        del self.app
        gc.collect()

    def singleShot(self, *args):
        QTimer.singleShot(*args)
