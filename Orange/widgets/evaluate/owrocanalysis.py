"""
ROC Analysis Widget
-------------------

"""
from collections import namedtuple, deque

import numpy

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import pyqtgraph as pg

import sklearn.metrics

import Orange.data
import Orange.evaluation.testing

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette


class OWROCAnalysis(widget.OWWidget):
    name = "ROC Analysis"
    description = ("Displays Receiver Operating Characteristics curve " +
                   "based on evaluation of classifiers.")
    icon = "icons/ROCAnalysis.svg"

    inputs = [
        {"name": "Evaluation Results",
         "type": Orange.evaluation.testing.Results,
         "handler": "set_results"}
    ]

    target_index = settings.Setting(0)
    selected_classifiers = []

    display_perf_line = settings.Setting(True)
    display_def_threshold = settings.Setting(True)

    fp_cost = settings.Setting(500)
    fn_cost = settings.Setting(500)
    target_prior = settings.Setting(50.0)

    #: ROC Averaging Types
    Merge, Vertical, Threshold, NoAveraging = 0, 1, 2, 3
    roc_averaging = settings.Setting(Merge)

    display_convex_hull = settings.Setting(False)
    display_convex_curve = settings.Setting(False)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.results = None
        self.classifier_names = []
        self.perf_line = None
        self.threshold_opints = []

        box = gui.widgetBox(self.controlArea, "Plot")
        tbox = gui.widgetBox(box, "Target Class")
        tbox.setFlat(True)

        self.target_cb = gui.comboBox(
            tbox, self, "target_index", callback=self._on_target_changed)

        cbox = gui.widgetBox(box, "Classifiers")
        cbox.setFlat(True)
        gui.listBox(cbox, self, "selected_classifiers", "classifier_names",
                    selectionMode=QtGui.QListView.MultiSelection,
                    callback=self._on_classifiers_changed)

        abox = gui.widgetBox(box, "Average ROC Curves")
        abox.setFlat(True)
        gui.comboBox(abox, self, "roc_averaging",
                     items=["Merge (expected ROC perf.)", "Vertical",
                            "Threshold", "None"],
                     callback=self._replot)

        hbox = gui.widgetBox(box, "ROC Convex Hull")
        hbox.setFlat(True)
#         gui.checkBox(hbox, self, "display_convex_hull",
#                      "Show ROC convex hull", callback=self._replot)
        gui.checkBox(hbox, self, "display_convex_curve",
                     "Show convex ROC curves", callback=self._replot)

        box = gui.widgetBox(self.controlArea, "Analysis")

#         gui.checkBox(box, self, "display_perf_line", "Show performance line",
#                      callback=self._on_display_perf_line_changed)
        gui.checkBox(box, self, "display_def_threshold",
                     "Default threshold (0.5) point",
                     callback=self._on_display_def_threshold_changed)

        self.plotview = pg.GraphicsView(background="w")
        self.plotview.setFrameStyle(QtGui.QFrame.StyledPanel)

        self.plot = pg.PlotItem()
        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))

        tickfont = QtGui.QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setTickFont(tickfont)
        axis.setPen(pen)
        axis.setLabel("FP Rate (1-Specificity)")

        axis = self.plot.getAxis("left")
        axis.setTickFont(tickfont)
        axis.setPen(pen)
        axis.setLabel("TP Rate (Sensitivity)")

        self.plot.showGrid(True, True, alpha=0.1)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))

        self.plotview.setCentralItem(self.plot)
        self.mainArea.layout().addWidget(self.plotview)

    def set_results(self, results):
        self.clear()
        self.error(0)

        if results is not None:
            if results.data is None:
                self.error(0, "Give me data!!")
                results = None
            elif not isinstance(results.data.domain.class_var,
                                Orange.data.DiscreteVariable):
                self.error(0, "Need discrete class variable")
                results = None

        self.results = results

        if results is not None:
            self._initialize(results)
            self._setup_plot()

    def clear(self):
        """Clear the widget state."""
        self.results = None
        self.plot.clear()
        self.classifier_names = []
        self.selected_classifiers = []
        self.target_cb.clear()
        self.target_index = 0

    def _initialize(self, results):
        names = getattr(results, "fitter_names", None)

        if names is None:
            names = ["#{}".format(i + 1)
                     for i in range(len(results.predicted))]

        self.classifier_names = names
        self.selected_classifiers = list(range(len(names)))

        class_var = results.data.domain.class_var
        self.target_cb.addItems(class_var.values)

    def _setup_plot(self):
        folds = self.results.folds
        actual = self.results.actual
        probs = self.results.probabilities
        target = self.target_index

        def generate_pens(basecolor):
            pen = QtGui.QPen(basecolor, 1)
            pen.setCosmetic(True)

            shadow_pen = QtGui.QPen(pen.color().lighter(160), 2.5)
            shadow_pen.setCosmetic(True)
            return pen, shadow_pen

        def generate_colors(ncolors):
            palette = colorpalette.ColorPaletteGenerator(ncolors)
            return [palette[i] for i in range(ncolors)]

        colors = generate_colors(len(self.classifier_names))

        if self.roc_averaging == OWROCAnalysis.Merge:
            for clf_idx in self.selected_classifiers:
                pen, shadow_pen = generate_pens(colors[clf_idx])
                fpr, tpr, thres = sklearn.metrics.roc_curve(
                    actual,
                    probs[clf_idx, :, target],
                    pos_label=target,
                )
                self.plot.plot(
                    fpr, tpr,
                    pen=pen, shadowPen=shadow_pen,
                    symbol="+", symbolSize=3, symbolPen=shadow_pen,
                    antialias=True,
                )
                if self.display_convex_curve:
                    hull = roc_curve_convex_hull((fpr, tpr, thres))
                    self.plot.plot(hull[0], hull[1], pen=pen, antialias=True)

                if self.display_def_threshold:
                    ind = numpy.argmin(numpy.abs(thres - 0.5))
                    item = pg.TextItem(
                        text="{:.3f}".format(thres[ind]),
                    )
                    item.setPos(fpr[ind], tpr[ind])
                    self.plot.addItem(item)

        elif self.roc_averaging == OWROCAnalysis.Vertical:
            for clf_idx in self.selected_classifiers:
                pen, shadow_pen = generate_pens(colors[clf_idx])
                curves = [roc_curve_for_fold(self.results, fold,
                                             clf_idx, target)
                          for fold in self.results.folds]

                fpr, tpr, std = roc_curve_vertical_average(curves)
                self.plot.plot(
                    fpr, tpr,
                    pen=pen, shadowPen=shadow_pen,
                    symbol="+", symbolSize=4, symbolPen=shadow_pen,
                    antialias=True,
                )
                N = len(curves)
                Z = 1.96
                confint = Z * std / numpy.sqrt(N)
                item = pg.ErrorBarItem(
                    x=fpr[1:-1], y=tpr[1: -1],
                    height=2 * confint[1:-1],
                    pen=pen, beam=0.025
                )
                self.plot.addItem(item)

        elif self.roc_averaging == OWROCAnalysis.Threshold:
            for clf_idx in self.selected_classifiers:
                pen, shadow_pen = generate_pens(colors[clf_idx])
                curves = [roc_curve_for_fold(self.results, fold,
                                             clf_idx, target)
                          for fold in self.results.folds]
                thres = numpy.linspace(1.0, 0.0, 10)
                all_thresh = numpy.hstack([t for _, _, t in curves])
                all_thresh = numpy.clip(all_thresh, 0.0, 1.0)
                all_thresh = numpy.unique(all_thresh)

                thres = all_thresh[::max(all_thresh.size // 10, 1)]
                (fpr, fpr_std), (tpr, tpr_std) = \
                    roc_curve_threshold_average(curves, thres)
                self.plot.plot(
                    fpr, tpr,
                    pen=pen, shadowPen=shadow_pen,
                    symbol="+", symbolSize=4, symbolPen=shadow_pen,
                    antialias=True,
                )
                N = len(curves)
                Z = 1.96
                fpr_confint = Z * fpr_std / numpy.sqrt(N)
                tpr_confint = Z * tpr_std / numpy.sqrt(N)

                item = pg.ErrorBarItem(
                    x=fpr[1:-1], y=tpr[1: -1],
                    width=2 * fpr_confint[1:-1], height=2 * tpr_confint[1:-1],
                    pen=pen, beam=0.025
                )
                self.plot.addItem(item)

        elif self.roc_averaging == OWROCAnalysis.NoAveraging:
            for clf_idx in self.selected_classifiers:
                pen, shadow_pen = generate_pens(colors[clf_idx])
                for fold in folds:
                    fpr, tpr, thres = roc_curve_for_fold(
                        self.results, fold, clf_idx, target
                    )
                    self.plot.plot(
                        fpr, tpr,
                        pen=pen, shadowPen=shadow_pen,
                        symbol="+", symbolSize=4, symbolPen=shadow_pen,
                        antialias=True,
                    )
                    if self.display_convex_curve:
                        hull = roc_curve_convex_hull((fpr, tpr, thres))
                        self.plot.plot(
                            hull[0], hull[1], pen=pen, antialias=True
                        )

        pen = QtGui.QPen(QtGui.QColor(100, 100, 100, 100), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.plot.plot([0, 1], [0, 1], pen=pen, antialias=True)

    def _on_target_changed(self):
        self.plot.clear()
        self._setup_plot()

    def _on_classifiers_changed(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()

    def _on_display_perf_line_changed(self):
        if self.perf_line is not None:
            self.perf_line.setVisible(self.display_perf_line)

    def _on_display_def_threshold_changed(self):
        self._replot()
#         if self.threshold_points:
#             for point in self.threshold_points:
#                 point.setVisible(self.display_def_threshold)

    def _replot(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()


def interp(x, xp, fp, left=None, right=None):
    """
    Like numpy.interp except for handling of running sequences of
    same values in `xp`.
    """
    x = numpy.asanyarray(x)
    xp = numpy.asanyarray(xp)
    fp = numpy.asanyarray(fp)

    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape")

    ind = numpy.searchsorted(xp, x, side="right")
    fx = numpy.zeros(len(x))

    under = ind == 0
    over = ind == len(xp)
    between = ~under & ~over

    fx[under] = left if left is not None else fp[0]
    fx[over] = right if right is not None else fp[-1]

    if right is not None:
        # Fix points exactly on the right boundary.
        fx[x == xp[-1]] = fp[-1]

    ind = ind[between]

    df = (fp[ind] - fp[ind - 1]) / (xp[ind] - xp[ind - 1])

    fx[between] = df * (x[between] - xp[ind]) + fp[ind]

    return fx


def roc_curve_for_fold(res, fold, clf_idx, target):
    fold_actual = res.actual[fold]
    fold_probs = res.probabilities[clf_idx][fold][:, target]
    return sklearn.metrics.roc_curve(
        fold_actual, fold_probs, pos_label=target
    )


def roc_curve_vertical_average(curves, samples=10):
    fpr_sample = numpy.linspace(0.0, 1.0, samples)
    tpr_samples = []
    for fpr, tpr, _ in curves:
        tpr_samples.append(interp(fpr_sample, fpr, tpr, left=0, right=1))

    tpr_samples = numpy.array(tpr_samples)
    return fpr_sample, tpr_samples.mean(axis=0), tpr_samples.std(axis=0)


def roc_curve_threshold_average(curves, thresh_samples):
    fpr_samples, tpr_samples = [], []
    for fpr, tpr, thresh in curves:
        ind = numpy.searchsorted(thresh[::-1], thresh_samples, side="left")
        ind = ind[::-1]
        ind = numpy.clip(ind, 0, len(thresh) - 1)
        fpr_samples.append(fpr[ind])
        tpr_samples.append(tpr[ind])

    fpr_samples = numpy.array(fpr_samples)
    tpr_samples = numpy.array(tpr_samples)

    return ((fpr_samples.mean(axis=0), fpr_samples.std(axis=0)),
            (tpr_samples.mean(axis=0), fpr_samples.std(axis=0)))


def roc_curve_threshold_average_interp(curves, thresh_samples):
    fpr_samples, tpr_samples = [], []
    for fpr, tpr, thresh in curves:
        thresh = thresh[::-1]
        fpr = interp(thresh_samples, thresh, fpr[::-1], left=1.0, right=0.0)
        tpr = interp(thresh_samples, thresh, tpr[::-1], left=1.0, right=0.0)
        fpr_samples.append(fpr)
        tpr_samples.append(tpr)

    fpr_samples = numpy.array(fpr_samples)
    tpr_samples = numpy.array(tpr_samples)

    return ((fpr_samples.mean(axis=0), fpr_samples.std(axis=0)),
            (tpr_samples.mean(axis=0), fpr_samples.std(axis=0)))


roc_point = namedtuple("roc_point", ["fpr", "tpr", "threshold"])


def roc_curve_convex_hull(curve):
    def slope(p1, p2):
        x1, y1, _ = p1
        x2, y2, _ = p2
        if x1 != x2:
            return  (y2 - y1) / (x2 - x1)
        else:
            return numpy.inf

    fpr, _, _ = curve

    if len(fpr) <= 2:
        return curve
    points = map(roc_point._make, zip(*curve))

    hull = deque([next(points)])

    for point in points:
        while True:
            if len(hull) < 2:
                hull.append(point)
                break
            else:
                last = hull[-1]
                if point.fpr != last.fpr and \
                        slope(hull[-2], last) > slope(last, point):
                    hull.append(point)
                    break
                else:
                    hull.pop()

    fpr = numpy.array([p.fpr for p in hull])
    tpr = numpy.array([p.tpr for p in hull])
    thres = numpy.array([p.threshold for p in hull])
    return (fpr, tpr, thres)


def main():
    import gc
    import sip
    from PyQt4.QtGui import QApplication
    from Orange.classification import logistic_regression, svm

    app = QApplication([])
    w = OWROCAnalysis()
    w.show()
    w.raise_()

#     data = Orange.data.Table("iris")
    data = Orange.data.Table("ionosphere")
    results = Orange.evaluation.testing.CrossValidation(
        data,
        [logistic_regression.LogisticRegressionLearner(),
         logistic_regression.LogisticRegressionLearner(penalty="l1"),
         svm.SVMLearner(probability=True),
         svm.NuSVMLearner(probability=True)],
        k=5,
        store_data=True,
    )
    results.fitter_names = ["Logistic", "Logistic (L1 reg.)", "SVM", "NuSVM"]
    w.set_results(results)

    rval = app.exec_()
    w.deleteLater()
    sip.delete(w)
    del w
    app.processEvents()
    sip.delete(app)
    del app
    gc.collect()
    return rval

if __name__ == "__main__":
    import sys
    sys.exit(main())
