import unittest
from unittest import mock
from importlib import import_module
import os
import warnings
import tempfile

import AnyQt
from AnyQt.QtGui import QFont, QBrush
from AnyQt.QtCore import Qt

from Orange.data.table import Table
from Orange.classification import LogisticRegressionLearner
from Orange.classification.tree import TreeLearner
from Orange.evaluation import CrossValidation
from Orange.distance import Euclidean
from Orange.canvas.report.owreport import OWReport
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.owtreeviewer import OWTreeGraph
from Orange.widgets.data.owfile import OWFile
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.owtestlearners import OWTestLearners
from Orange.widgets.unsupervised.owcorrespondence import OWCorrespondenceAnalysis
from Orange.widgets.unsupervised.owdistancemap import OWDistanceMap
from Orange.widgets.unsupervised.owdistances import OWDistances
from Orange.widgets.unsupervised.owhierarchicalclustering import OWHierarchicalClustering
from Orange.widgets.unsupervised.owkmeans import OWKMeans
from Orange.widgets.unsupervised.owmds import OWMDS
from Orange.widgets.unsupervised.owpca import OWPCA
from Orange.widgets.utils.itemmodels import PyTableModel


def get_owwidgets(top_module_name):
    top_module = import_module(top_module_name)
    widgets = []
    for root, _, files in os.walk(top_module.__path__[0]):
        root = root[len(top_module.__path__[0]):].lstrip(os.path.sep)
        for file in files:
            if file.lower().startswith('ow') and file.lower().endswith('.py'):
                module_name = "{}.{}".format(
                    top_module_name,
                    os.path.join(root, file).replace(os.path.sep, '.')[:-len('.py')])
                try:
                    module = import_module(module_name,
                                           top_module_name[:top_module_name.index('.')])
                except (ImportError, RuntimeError):
                    warnings.warn('Failed to import module: ' + module_name)
                    continue
                for name, value in module.__dict__.items():
                    if (name.upper().startswith('OW') and
                            isinstance(value, type) and
                            issubclass(value, OWWidget) and
                            getattr(value, 'name', None) and
                            getattr(value, 'send_report', None)):
                        widgets.append(value)
    return list(set(widgets))


DATA_WIDGETS = get_owwidgets('Orange.widgets.data')
VISUALIZATION_WIDGETS = get_owwidgets('Orange.widgets.visualize')
MODEL_WIDGETS = get_owwidgets('Orange.widgets.model')


class TestReport(WidgetTest):
    def test_report(self):
        count = 5
        for _ in range(count):
            rep = OWReport.get_instance()
            file = self.create_widget(OWFile)
            file.create_report_html()
            rep.make_report(file)
        self.assertEqual(rep.table_model.rowCount(), count)

    def test_report_table(self):
        rep = OWReport.get_instance()
        model = PyTableModel([['x', 1, 2],
                              ['y', 2, 2]])
        model.setHorizontalHeaderLabels(['a', 'b', 'c'])

        model.setData(model.index(0, 0), Qt.AlignHCenter | Qt.AlignTop, Qt.TextAlignmentRole)
        model.setData(model.index(1, 0), QFont('', -1, QFont.Bold), Qt.FontRole)
        model.setData(model.index(1, 2), QBrush(Qt.red), Qt.BackgroundRole)

        view = gui.TableView()
        view.show()
        view.setModel(model)
        rep.report_table('Name', view)
        self.maxDiff = None
        self.assertEqual(
            rep.report_html,
            '<h2>Name</h2><table>\n'
            '<tr>'
            '<th style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:left;vertical-align:middle;">a</th>'
            '<th style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:left;vertical-align:middle;">b</th>'
            '<th style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:left;vertical-align:middle;">c</th>'
            '</tr>'
            '<tr>'
            '<td style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:center;vertical-align:top;">x</td>'
            '<td style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:right;vertical-align:middle;">1</td>'
            '<td style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:right;vertical-align:middle;">2</td>'
            '</tr>'
            '<tr>'
            '<td style="color:black;border:0;background:transparent;'
            'font-weight:bold;text-align:left;vertical-align:middle;">y</td>'
            '<td style="color:black;border:0;background:transparent;'
            'font-weight:normal;text-align:right;vertical-align:middle;">2</td>'
            '<td style="color:black;border:0;background:#ff0000;'
            'font-weight:normal;text-align:right;vertical-align:middle;">2</td>'
            '</tr></table>')

    def test_save_report_permission(self):
        """
        Permission Error may occur when trying to save report.
        GH-2147
        """
        rep = OWReport.get_instance()
        patch_target_1 = "Orange.canvas.report.owreport.open"
        patch_target_2 = "AnyQt.QtWidgets.QFileDialog.getSaveFileName"
        patch_target_3 = "AnyQt.QtWidgets.QMessageBox.exec_"
        filenames = ["f.report", "f.html"]
        for filename in filenames:
            with unittest.mock.patch(patch_target_1, create=True, side_effect=PermissionError),\
                    unittest.mock.patch(patch_target_2, return_value=(filename, 0)),\
                    unittest.mock.patch(patch_target_3, return_value=True):
                rep.save_report()

    def test_save_report(self):
        rep = OWReport.get_instance()
        file = self.create_widget(OWFile)
        file.create_report_html()
        rep.make_report(file)
        temp_dir = tempfile.mkdtemp()
        temp_name = os.path.join(temp_dir, "f.report")
        try:
            with mock.patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName",
                            return_value=(temp_name, 0)), \
                    mock.patch("AnyQt.QtWidgets.QMessageBox.exec_",
                               return_value=True):
                rep.save_report()
        finally:
            os.remove(temp_name)
            os.rmdir(temp_dir)


class TestReportWidgets(WidgetTest):
    model_widgets = MODEL_WIDGETS
    data_widgets = DATA_WIDGETS
    eval_widgets = [OWCalibrationPlot, OWLiftCurve, OWROCAnalysis]
    unsu_widgets = [OWCorrespondenceAnalysis, OWDistances, OWKMeans,
                    OWMDS, OWPCA]
    dist_widgets = [OWDistanceMap, OWHierarchicalClustering]
    visu_widgets = VISUALIZATION_WIDGETS
    spec_widgets = [OWTestLearners, OWTreeGraph]

    def _create_report(self, widgets, rep, data):
        for widget in widgets:
            w = self.create_widget(widget)
            if w.inputs and isinstance(data, w.inputs[0].type):
                handler = getattr(w, w.inputs[0].handler)
                handler(data)
                w.create_report_html()
            rep.make_report(w)
            # rep.show()

    def test_report_widgets_model(self):
        rep = OWReport.get_instance()
        data = Table("titanic")
        widgets = self.model_widgets

        w = self.create_widget(OWTreeGraph)
        clf = TreeLearner(max_depth=3)(data)
        clf.instances = data
        w.ctree(clf)
        w.create_report_html()
        rep.make_report(w)

        self._create_report(widgets, rep, data)

    def test_report_widgets_data(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.data_widgets
        self._create_report(widgets, rep, data)

    def test_report_widgets_evaluate(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.eval_widgets
        results = CrossValidation(data,
                                  [LogisticRegressionLearner()],
                                  store_data=True)
        results.learner_names = ["LR l2"]

        w = self.create_widget(OWTestLearners)
        set_learner = getattr(w, w.Inputs.learner.handler)
        set_train = getattr(w, w.Inputs.train_data.handler)
        set_test = getattr(w, w.Inputs.test_data.handler)
        set_learner(LogisticRegressionLearner(), 0)
        set_train(data)
        set_test(data)
        w.create_report_html()
        rep.make_report(w)

        self._create_report(widgets, rep, results)

    def test_report_widgets_unsupervised(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.unsu_widgets
        self._create_report(widgets, rep, data)

    def test_report_widgets_unsupervised_dist(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        dist = Euclidean(data)
        widgets = self.dist_widgets
        self._create_report(widgets, rep, dist)

    def test_report_widgets_visualize(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.visu_widgets
        self._create_report(widgets, rep, data)

    @unittest.skipIf(AnyQt.USED_API == "pyqt5", "Segfaults on PyQt5")
    def test_report_widgets_all(self):
        rep = OWReport.get_instance()
        widgets = self.model_widgets + self.data_widgets + self.eval_widgets + \
                  self.unsu_widgets + self.dist_widgets + self.visu_widgets + \
                  self.spec_widgets
        self._create_report(widgets, rep, None)


if __name__ == "__main__":
    unittest.main()
