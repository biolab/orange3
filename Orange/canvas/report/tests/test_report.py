import unittest
import pickle
import sys
from PyQt4.QtGui import QApplication
from Orange.canvas.report.owreport import OWReport
from Orange.widgets.data.owfile import OWFile


class TestReport(unittest.TestCase):
    def test_report(self):
        count = 5
        app = QApplication(sys.argv)
        for i in range(count):
            rep = OWReport.get_instance()
            file = OWFile()
            file.create_report_html()
            rep.make_report(file)
        self.assertEqual(rep.table_model.rowCount(), count)

    def test_report_pickle(self):
        app = QApplication(sys.argv)
        rep = OWReport().get_instance()
        p = pickle.dumps(rep)
        rep2 = pickle.loads(p)
        self.assertEqual(type(rep), type(rep2))
