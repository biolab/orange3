import unittest
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
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.classify.owclassificationtree import OWClassificationTree
from Orange.widgets.classify.owclassificationtreegraph import OWClassificationTreeGraph
from Orange.widgets.classify.owknn import OWKNNLearner
from Orange.widgets.classify.owlogisticregression import OWLogisticRegression
from Orange.widgets.classify.owmajority import OWMajority
from Orange.widgets.classify.ownaivebayes import OWNaiveBayes
from Orange.widgets.classify.owrandomforest import OWRandomForest
from Orange.widgets.classify.owsvmclassification import OWSVMClassification
from Orange.widgets.data.owconcatenate import OWConcatenate
from Orange.widgets.data.owcontinuize import OWContinuize
from Orange.widgets.data.owdatainfo import OWDataInfo
from Orange.widgets.data.owdatasampler import OWDataSampler
from Orange.widgets.data.owdiscretize import OWDiscretize
from Orange.widgets.data.owfeatureconstructor import OWFeatureConstructor
from Orange.widgets.data.owfile import OWFile
from Orange.widgets.data.owimpute import OWImpute
from Orange.widgets.data.owmergedata import OWMergeData
from Orange.widgets.data.owoutliers import OWOutliers
from Orange.widgets.data.owpaintdata import OWPaintData
from Orange.widgets.data.owpurgedomain import OWPurgeDomain
from Orange.widgets.data.owrank import OWRank
from Orange.widgets.data.owselectcolumns import OWSelectAttributes
from Orange.widgets.data.owselectrows import OWSelectRows
from Orange.widgets.data.owsql import OWSql
from Orange.widgets.data.owtable import OWDataTable
from Orange.widgets.data.owcolor import OWColor
from Orange.widgets.data.owpreprocess import OWPreprocess
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.owtestlearners import OWTestLearners
from Orange.widgets.regression.owregressiontree import OWRegressionTree
from Orange.widgets.regression.owregressiontreegraph import OWRegressionTreeGraph
from Orange.widgets.regression.owknnregression import OWKNNRegression
from Orange.widgets.regression.owmean import OWMean
from Orange.widgets.regression.owsvmregression import OWSVMRegression
from Orange.widgets.unsupervised.owcorrespondence import OWCorrespondenceAnalysis
from Orange.widgets.unsupervised.owdistancemap import OWDistanceMap
from Orange.widgets.unsupervised.owdistances import OWDistances
from Orange.widgets.unsupervised.owhierarchicalclustering import OWHierarchicalClustering
from Orange.widgets.unsupervised.owkmeans import OWKMeans
from Orange.widgets.unsupervised.owmds import OWMDS
from Orange.widgets.unsupervised.owpca import OWPCA
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.visualize.owboxplot import OWBoxPlot
from Orange.widgets.visualize.owdistributions import OWDistributions
from Orange.widgets.visualize.owheatmap import OWHeatMap
from Orange.widgets.visualize.owlinearprojection import OWLinearProjection
from Orange.widgets.visualize.owmosaic import OWMosaicDisplay
from Orange.widgets.visualize.owscattermap import OWScatterMap
from Orange.widgets.visualize.owscatterplot import OWScatterPlot
from Orange.widgets.visualize.owsieve import OWSieveDiagram
from Orange.widgets.visualize.owvenndiagram import OWVennDiagram


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


@unittest.skip('Segfaults. Dunno. @astaric says it might be something on the QWidget.')
class TestReportWidgets(WidgetTest):
    clas_widgets = [OWClassificationTree, OWKNNLearner, OWLogisticRegression,
                    OWMajority, OWNaiveBayes, OWRandomForest,
                    OWSVMClassification]
    data_widgets = [OWConcatenate, OWContinuize, OWDataInfo, OWDataSampler,
                    OWDiscretize, OWFeatureConstructor, OWOutliers, OWImpute,
                    OWMergeData, OWFile, OWPaintData, OWPurgeDomain, OWRank,
                    OWSelectAttributes, OWSelectRows, OWSql, OWDataTable,
                    OWColor, OWPreprocess]
    eval_widgets = [OWCalibrationPlot, OWLiftCurve, OWROCAnalysis]
    regr_widgets = [OWRegressionTree, OWKNNRegression, OWMean, OWSVMRegression]
    unsu_widgets = [OWCorrespondenceAnalysis, OWDistances, OWKMeans,
                    OWMDS, OWPCA]
    dist_widgets = [OWDistanceMap, OWHierarchicalClustering]
    visu_widgets = [OWBoxPlot, OWDistributions, OWHeatMap, OWLinearProjection,
                    OWMosaicDisplay, OWScatterPlot,
                    OWSieveDiagram, OWScatterMap, OWVennDiagram]
    spec_widgets = [OWClassificationTreeGraph, OWTestLearners,
                    OWRegressionTreeGraph]

    def _create_report(self, widgets, rep, data):
        for widget in widgets:
            w = widget()
            if w.inputs and data is not None:
                handler = getattr(w, w.inputs[0].handler)
                handler(data)
            w.create_report_html()
            rep.make_report(w)
            # rep.show()

    def test_report_widgets_classify(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.clas_widgets

        w = OWClassificationTreeGraph()
        clf = TreeLearner(max_depth=3)(data)
        clf.instances = data
        w.ctree(clf)
        w.create_report_html()
        rep.make_report(w)

        self.assertEqual(len(widgets) + 1, 8)
        self._create_report(widgets, rep, data)

    def test_report_widgets_data(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.data_widgets
        self.assertEqual(len(widgets), 19)
        self._create_report(widgets, rep, data)

    def test_report_widgets_evaluate(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.eval_widgets
        results = CrossValidation(data,
                                  [LogisticRegressionLearner()],
                                  store_data=True)
        results.learner_names = ["LR l2"]

        w = OWTestLearners()
        set_learner = getattr(w, w.inputs[0].handler)
        set_train = getattr(w, w.inputs[1].handler)
        set_test = getattr(w, w.inputs[2].handler)
        set_learner(LogisticRegressionLearner(), 0)
        set_train(data)
        set_test(data)
        w.create_report_html()
        rep.make_report(w)

        self.assertEqual(len(widgets) + 1, 4)
        self._create_report(widgets, rep, results)

    def test_report_widgets_regression(self):
        rep = OWReport.get_instance()
        data = Table("housing")
        widgets = self.regr_widgets

        w = OWRegressionTreeGraph()
        mod = TreeRegressionLearner(max_depth=3)(data)
        mod.instances = data
        w.ctree(mod)
        w.create_report_html()
        rep.make_report(w)

        self.assertEqual(len(widgets) + 1, 5)
        self._create_report(widgets, rep, data)

    def test_report_widgets_unsupervised(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.unsu_widgets
        self.assertEqual(len(widgets), 5)
        self._create_report(widgets, rep, data)

    def test_report_widgets_unsupervised_dist(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        dist = Euclidean(data)
        widgets = self.dist_widgets
        self.assertEqual(len(widgets), 2)
        self._create_report(widgets, rep, dist)

    def test_report_widgets_visualize(self):
        rep = OWReport.get_instance()
        data = Table("zoo")
        widgets = self.visu_widgets
        self.assertEqual(len(widgets), 9)
        self._create_report(widgets, rep, data)

    @unittest.skip('... in favor of other methods. Segfaults.')
    def test_report_widgets_all(self):
        rep = OWReport.get_instance()
        widgets = self.clas_widgets + self.data_widgets + self.eval_widgets + \
                  self.regr_widgets + self.unsu_widgets + self.dist_widgets + \
                  self.visu_widgets + self.spec_widgets
        self.assertEqual(len(widgets), 52)
        self._create_report(widgets, rep, None)
