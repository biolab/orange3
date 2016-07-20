from importlib import import_module
import os
import warnings

from PyQt4.QtGui import QFont, QBrush
from PyQt4.QtCore import Qt
from Orange.data.table import Table
from Orange.classification import LogisticRegressionLearner
from Orange.classification.tree import TreeLearner
from Orange.regression.tree import TreeRegressionLearner
from Orange.evaluation import CrossValidation
from Orange.distance import Euclidean
from Orange.canvas.report.owreport import OWReport
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.classify.owclassificationtreegraph import OWClassificationTreeGraph
from Orange.widgets.data.owfile import OWFile
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.owtestlearners import OWTestLearners
from Orange.widgets.regression.owregressiontreegraph import OWRegressionTreeGraph
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
    for root, dirs, files in os.walk(top_module.__path__[0]):
        root = root[len(top_module.__path__[0]):].lstrip(os.path.sep)
        for file in files:
            if file.lower().startswith('ow') and file.lower().endswith('.py'):
                module_name = top_module_name + '.' + os.path.join(root, file).replace(os.path.sep, '.')[:-len('.py')]
                try:
                    module = import_module(module_name, top_module_name[:top_module_name.index('.')])
                except ImportError:
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
CLASSIFICATION_WIDGETS = get_owwidgets('Orange.widgets.classify')
REGRESSION_WIDGETS = get_owwidgets('Orange.widgets.regression')


class TestReport(WidgetTest):
    def test_report(self):
        count = 5
        for i in range(count):
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


class TestReportWidgets(WidgetTest):
    clas_widgets = CLASSIFICATION_WIDGETS
    data_widgets = DATA_WIDGETS
    eval_widgets = [OWCalibrationPlot, OWLiftCurve, OWROCAnalysis]
    regr_widgets = REGRESSION_WIDGETS
    unsu_widgets = [OWCorrespondenceAnalysis, OWDistances, OWKMeans,
                    OWMDS, OWPCA]
    dist_widgets = [OWDistanceMap, OWHierarchicalClustering]
    visu_widgets = VISUALIZATION_WIDGETS
    spec_widgets = [OWClassificationTreeGraph, OWTestLearners,
                    OWRegressionTreeGraph]

    def _create_report(self, widgets, rep, data):
        for widget in widgets:
            w = self.create_widget(widget)
            if w.inputs and isinstance(data, w.inputs[0].type):
                handler = getattr(w, w.inputs[0].handler)
                handler(data)
                w.create_report_html()
            rep.make_report(w)
            # rep.show()

    def test_report_widgets_classify(self):
        rep = OWReport.get_instance()
        data = Table("titanic")
        widgets = self.clas_widgets

        w = self.create_widget(OWClassificationTreeGraph)
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
        set_learner = getattr(w, w.inputs[0].handler)
        set_train = getattr(w, w.inputs[1].handler)
        set_test = getattr(w, w.inputs[2].handler)
        set_learner(LogisticRegressionLearner(), 0)
        set_train(data)
        set_test(data)
        w.create_report_html()
        rep.make_report(w)

        self._create_report(widgets, rep, results)

    def test_report_widgets_regression(self):
        rep = OWReport.get_instance()
        data = Table("housing")
        widgets = self.regr_widgets

        w = self.create_widget(OWRegressionTreeGraph)
        mod = TreeRegressionLearner(max_depth=3)(data)
        mod.instances = data
        w.ctree(mod)
        w.create_report_html()
        rep.make_report(w)

        self._create_report(widgets, rep, data)

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

    def test_report_widgets_all(self):
        rep = OWReport.get_instance()
        widgets = self.clas_widgets + self.data_widgets + self.eval_widgets + \
                  self.regr_widgets + self.unsu_widgets + self.dist_widgets + \
                  self.visu_widgets + self.spec_widgets
        self._create_report(widgets, rep, None)
