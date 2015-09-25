"""
Calibration Plot Widget
-----------------------

"""
from collections import namedtuple

from PyQt4 import QtGui

import pyqtgraph as pg

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette, colorbrewer
from Orange.widgets.io import FileFormats


Curve = namedtuple(
    "Curve",
    ["x", "y"]
)

PlotCurve = namedtuple(
    "PlotCurve",
    ["curve",
     "curve_item",
     "rug_item"]
)


class OWCalibrationPlot(widget.OWWidget):
    name = "Calibration Plot"
    description = "Calibration plot based on evaluation of classifiers."
    icon = "icons/CalibrationPlot.svg"
    priority = 1030
    inputs = [("Evaluation Results", Orange.evaluation.Results, "set_results")]

    target_index = settings.Setting(0)
    selected_classifiers = settings.Setting([])
    display_rug = settings.Setting(True)

    want_graph = True

    def __init__(self, parent=None):
        super().__init__(parent)

        self.results = None
        self.classifier_names = []
        self.colors = []
        self._curve_data = {}

        box = gui.widgetBox(self.controlArea, "Plot")
        tbox = gui.widgetBox(box, "Target Class")
        tbox.setFlat(True)

        self.target_cb = gui.comboBox(
            tbox, self, "target_index", callback=self._replot,
            contentsLength=8)

        cbox = gui.widgetBox(box, "Classifier")
        cbox.setFlat(True)

        self.classifiers_list_box = gui.listBox(
            box, self, "selected_classifiers", "classifier_names",
            selectionMode=QtGui.QListWidget.MultiSelection,
            callback=self._replot)

        gui.checkBox(box, self, "display_rug", "Show rug",
                     callback=self._on_display_rug_changed)

        self.plotview = pg.GraphicsView(background="w")
        self.plot = pg.PlotItem()

        axis = self.plot.getAxis("bottom")
        axis.setLabel("Predicted Probability")

        axis = self.plot.getAxis("left")
        axis.setLabel("Observed Average")
        self.plotview.setCentralItem(self.plot)

        self.mainArea.layout().addWidget(self.plotview)
        self.graphButton.clicked.connect(self.save_graph)

    def set_results(self, results):
        self.clear()
        self.results = results

        if results is not None:
            self._initialize(results)
            self._replot()

    def clear(self):
        self.plot.clear()
        self.results = None
        self.classifier_names = []
        self.selected_classifiers = []
        self.target_cb.clear()
        self.target_index = 0
        self.colors = []
        self._curve_data = {}

    def _initialize(self, results):
        N = len(results.predicted)
        names = getattr(results, "learner_names", None)
        if names is None:
            names = ["#{}".format(i + 1) for i in range(N)]

        self.classifier_names = names
        self.colors = colorpalette.ColorPaletteGenerator(
            N, colorbrewer.colorSchemes["qualitative"]["Dark2"])

        for i in range(N):
            item = self.classifiers_list_box.item(i)
            item.setIcon(colorpalette.ColorPixmap(self.colors[i]))

        self.selected_classifiers = list(range(N))
        self.target_cb.addItems(results.data.domain.class_var.values)

    def plot_curve(self, clf_idx, target):
        if (clf_idx, target) in self._curve_data:
            return self._curve_data[clf_idx, target]

        ytrue = self.results.actual == target
        probs = self.results.probabilities[clf_idx, :, target]
        sortind = numpy.argsort(probs)
        probs = probs[sortind]
        ytrue = ytrue[sortind]
#         x = numpy.unique(probs)
        xmin, xmax = probs.min(), probs.max()
        x = numpy.linspace(xmin, xmax, 100)
        f = gaussian_smoother(probs, ytrue, sigma=0.15 * (xmax - xmin))
        observed = f(x)
        curve = Curve(x, observed)
        curve_item = pg.PlotDataItem(
            x, observed, pen=pg.mkPen(self.colors[clf_idx], width=1),
            shadowPen=pg.mkPen(self.colors[clf_idx].lighter(160), width=2),
            symbol="+", symbolSize=4,
            antialias=True
        )

        rh = 0.025
        rug_x = numpy.c_[probs, probs]
        rug_x_true = rug_x[ytrue].ravel()
        rug_x_false = rug_x[~ytrue].ravel()

        rug_y_true = numpy.ones_like(rug_x_true)
        rug_y_true[1::2] = 1 - rh
        rug_y_false = numpy.zeros_like(rug_x_false)
        rug_y_false[1::2] = rh

        rug1 = pg.PlotDataItem(
            rug_x_false, rug_y_false, pen=self.colors[clf_idx],
            connect="pairs", antialias=True
        )
        rug2 = pg.PlotDataItem(
            rug_x_true, rug_y_true, pen=self.colors[clf_idx],
            connect="pairs", antialias=True
        )
        self._curve_data[clf_idx, target] = PlotCurve(curve, curve_item, (rug1, rug2))
        return self._curve_data[clf_idx, target]

    def _setup_plot(self):
        target = self.target_index
        selected = self.selected_classifiers
        curves = [self.plot_curve(i, target) for i in selected]

        for curve in curves:
            self.plot.addItem(curve.curve_item)
            if self.display_rug:
                self.plot.addItem(curve.rug_item[0])
                self.plot.addItem(curve.rug_item[1])

        self.plot.plot([0, 1], [0, 1], antialias=True)

    def _replot(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()

    def _on_display_rug_changed(self):
        self._replot()

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.plot,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


import numpy


def gaussian_smoother(x, y, sigma=1.0):
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    gamma = 1. / (2 * sigma ** 2)
    a = 1. / (sigma * numpy.sqrt(2 * numpy.pi))

    if x.shape != y.shape:
        raise ValueError

    def smoother(xs):
        W = a * numpy.exp(-gamma * ((xs - x) ** 2))
        return numpy.average(y, weights=W)

    return numpy.frompyfunc(smoother, 1, 1)


def main():
    import sip
    from PyQt4.QtGui import QApplication
    from Orange.classification import (LogisticRegressionLearner, SVMLearner,
                                       NuSVMLearner)

    app = QApplication([])
    w = OWCalibrationPlot()
    w.show()
    w.raise_()

    data = Orange.data.Table("ionosphere")
    results = Orange.evaluation.CrossValidation(
        data,
        [LogisticRegressionLearner(penalty="l2"),
         LogisticRegressionLearner(penalty="l1"),
         SVMLearner(probability=True),
         NuSVMLearner(probability=True)
         ],
        store_data=True
    )
    results.learner_names = ["LR l2", "LR l1", "SVM", "Nu SVM"]
    w.set_results(results)
    rval = app.exec_()

    sip.delete(w)
    del w
    app.processEvents()
    del app
    return rval


if __name__ == "__main__":
    main()
