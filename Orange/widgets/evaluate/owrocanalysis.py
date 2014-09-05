"""
ROC Analysis Widget
-------------------

"""
import operator
from functools import reduce, wraps
from collections import namedtuple, deque

import numpy

from PyQt4 import QtGui
from PyQt4.QtGui import QColor, QPen, QBrush
from PyQt4.QtCore import Qt

import pyqtgraph as pg

import sklearn.metrics

import Orange.data
import Orange.evaluation.testing

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette, colorbrewer

#: Points on a ROC curve
ROCPoints = namedtuple(
    "ROCPoints",
    ["fpr",        # (N,) array of false positive rate coordinates (ascending)
     "tpr",        # (N,) array of true positive rate coordinates
     "thresholds"  # (N,) array of thresholds (in descending order)
     ]
)
ROCPoints.is_valid = property(lambda self: self.fpr.size > 0)

#: ROC Curve and it's convex hull
ROCCurve = namedtuple(
    "ROCCurve",
    ["points",  # ROCPoints
     "hull"     # ROCPoints of the convex hull
    ]
)
ROCCurve.is_valid = property(lambda self: self.points.is_valid)

#: A ROC Curve averaged vertically
ROCAveragedVert = namedtuple(
    "ROCAveragedVert",
    ["points",   # ROCPoints sampled by fpr
     "hull",     # ROCPoints of the convex hull
     "tpr_std",  # array standard deviation of tpr at each fpr point
     ]
)
ROCAveragedVert.is_valid = property(lambda self: self.points.is_valid)

#: A ROC Curve averaged by thresholds
ROCAveragedThresh = namedtuple(
    "ROCAveragedThresh",
    ["points",   # ROCPoints sampled by threshold
     "hull",     # ROCPoints of the convex hull
     "tpr_std",  # array standard deviations of tpr at each threshold
     "fpr_std"   # array standard deviations of fpr at each threshold
     ]
)
ROCAveragedThresh.is_valid = property(lambda self: self.points.is_valid)

#: Combined data for a ROC curve of a single algorithm
ROCData = namedtuple(
    "ROCData",
    ["merged",  # ROCCurve merged over all folds
     "folds",   # ROCCurve list, one for each fold
     "avg_vertical",   # ROCAveragedVert
     "avg_threshold",  # ROCAveragedThresh
     ]
)


def ROCData_from_results(results, clf_index, target):
    """
    Compute ROC Curve(s) from evaluation results.

    :param Orange.evaluation.Results results:
        Evaluation results.
    :param int clf_index:
        Learner/Fitter index in the `results`.
    :param int target:
        Target class index (i.e. positive class).
    :rval ROCData:
        A instance holding the computed curves.
    """
    merged = roc_curve_for_fold(results, slice(0, -1), clf_index, target)
    merged_curve = ROCCurve(ROCPoints(*merged),
                            ROCPoints(*roc_curve_convex_hull(merged)))

    folds = results.folds if results.folds is not None else [slice(0, -1)]
    fold_curves = []
    for fold in folds:
        # TODO: Check for no FP or no TP
        points = roc_curve_for_fold(results, fold, clf_index, target)
        hull = roc_curve_convex_hull(points)
        c = ROCCurve(ROCPoints(*points), ROCPoints(*hull))
        fold_curves.append(c)

    curves = [fold.points for fold in fold_curves
              if fold.is_valid]

    fpr, tpr, std = roc_curve_vertical_average(curves)
    thresh = numpy.zeros_like(fpr) * numpy.nan
    hull = roc_curve_convex_hull((fpr, tpr, thresh))
    v_avg = ROCAveragedVert(
        ROCPoints(fpr, tpr, thresh),
        ROCPoints(*hull),
        std
    )

    all_thresh = numpy.hstack([t for _, _, t in curves])
    all_thresh = numpy.clip(all_thresh, 0.0 - 1e-10, 1.0 + 1e-10)
    all_thresh = numpy.unique(all_thresh)[::-1]
    thresh = all_thresh[::max(all_thresh.size // 10, 1)]

    (fpr, fpr_std), (tpr, tpr_std) = \
        roc_curve_threshold_average(curves, thresh)

    hull = roc_curve_convex_hull((fpr, tpr, thresh))

    t_avg = ROCAveragedThresh(
        ROCPoints(fpr, tpr, thresh),
        ROCPoints(*hull),
        tpr_std,
        fpr_std
    )

    return ROCData(merged_curve, fold_curves, v_avg, t_avg)

ROCData.from_results = staticmethod(ROCData_from_results)

#: A curve item to be displayed in a plot
PlotCurve = namedtuple(
    "PlotCurve",
    ["curve",        # ROCCurve source curve
     "curve_item",   # pg.PlotDataItem main curve
     "hull_item"     # pg.PlotDataItem curve's convex hull
     ]
)


def plot_curve(curve, pen=None, shadow_pen=None, symbol="+",
               symbol_size=3, name=None):
    """
    Construct a `PlotCurve` for the given `ROCCurve`.

    :param ROCCurve curve:
        Source curve.

    The other parameters are passed to pg.PlotDataItem

    :rtype: PlotCurve
    """
    points = curve.points
    item = pg.PlotDataItem(
        points.fpr, points.tpr,
        pen=pen, shadowPen=shadow_pen,
        symbol=symbol, symbolSize=symbol_size, symbolPen=shadow_pen,
        name=name, antialias=True,
    )
    hull = curve.hull
    hull_item = pg.PlotDataItem(
        hull.fpr, hull.tpr, pen=pen, antialias=True
    )
    return PlotCurve(curve, item, hull_item)

PlotCurve.from_roc_curve = staticmethod(plot_curve)

#: A curve displayed in a plot with error bars
PlotAvgCurve = namedtuple(
    "PlotAvgCurve",
    ["curve",         # ROCCurve
     "curve_item",    # pg.PlotDataItem
     "hull_item",     # pg.PlotDataItem
     "confint_item",  # pg.ErrorBarItem
     ]
)


def plot_avg_curve(curve, pen=None, shadow_pen=None, symbol="+",
                   symbol_size=4, name=None):
    """
    Construct a `PlotAvgCurve` for the given `curve`.

    :param curve: Source curve.
    :type curve: ROCAveragedVert or ROCAveragedThresh

    The other parameters are passed to pg.PlotDataItem

    :rtype: PlotAvgCurve
    """
    pc = plot_curve(curve, pen=pen, shadow_pen=shadow_pen, symbol=symbol,
                    symbol_size=symbol_size, name=name)

    points = curve.points
    if isinstance(curve, ROCAveragedVert):
        tpr_std = curve.tpr_std
        error_item = pg.ErrorBarItem(
            x=points.fpr[1:-1], y=points.tpr[1:-1],
            height=2 * tpr_std[1:-1],
            pen=pen, beam=0.025,
            antialias=True,
        )
    elif isinstance(curve, ROCAveragedThresh):
        tpr_std, fpr_std = curve.tpr_std, curve.fpr_std
        error_item = pg.ErrorBarItem(
            x=points.fpr[1:-1], y=points.tpr[1:-1],
            height=2 * tpr_std[1:-1], width=2 * fpr_std[1:-1],
            pen=pen, beam=0.025,
            antialias=True,
        )
    return PlotAvgCurve(curve, pc.curve_item, pc.hull_item, error_item)

PlotAvgCurve.from_roc_curve = staticmethod(plot_avg_curve)

Some = namedtuple("Some", ["val"])


def once(f):
    """
    Return a function that will be called only once, and it's result cached.
    """
    cached = None

    @wraps(f)
    def wraped():
        nonlocal cached
        if cached is None:
            cached = Some(f())
        return cached.val
    return wraped


plot_curves = namedtuple(
    "plot_curves",
    ["merge",   # :: () -> PlotCurve
     "folds",   # :: () -> [PlotCurve]
     "avg_vertical",   # :: () -> PlotAvgCurve
     "avg_threshold",  # :: () -> PlotAvgCurve
     ]
)


class InfiniteLine(pg.InfiniteLine):
    """pyqtgraph.InfiniteLine extended to support antialiasing.
    """
    def __init__(self, pos=None, angle=90, pen=None, movable=False,
                 bounds=None, antialias=False):
        super().__init__(pos, angle, pen, movable, bounds)
        self.antialias = antialias

    def paint(self, painter, *args):
        if self.antialias:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        super().paint(painter, *args)


class OWROCAnalysis(widget.OWWidget):
    name = "ROC Analysis"
    description = ("Displays Receiver Operating Characteristics curve " +
                   "based on evaluation of classifiers.")
    icon = "icons/ROCAnalysis.svg"
    priority = 1010

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
        self.colors = []
        self._curve_data = {}
        self._plot_curves = {}
        self._rocch = None
        self._perf_line = None

        box = gui.widgetBox(self.controlArea, "Plot")
        tbox = gui.widgetBox(box, "Target Class")
        tbox.setFlat(True)

        self.target_cb = gui.comboBox(
            tbox, self, "target_index", callback=self._on_target_changed)

        cbox = gui.widgetBox(box, "Classifiers")
        cbox.setFlat(True)
        self.classifiers_list_box = gui.listBox(
            cbox, self, "selected_classifiers", "classifier_names",
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
        gui.checkBox(hbox, self, "display_convex_curve",
                     "Show convex ROC curves", callback=self._replot)
        gui.checkBox(hbox, self, "display_convex_hull",
                     "Show ROC convex hull", callback=self._replot)

        box = gui.widgetBox(self.controlArea, "Analysis")

        gui.checkBox(box, self, "display_def_threshold",
                     "Default threshold (0.5) point",
                     callback=self._on_display_def_threshold_changed)

        gui.checkBox(box, self, "display_perf_line", "Show performance line",
                     callback=self._on_display_perf_line_changed)
        grid = QtGui.QGridLayout()
        ibox = gui.indentedBox(box, orientation=grid)

        sp = gui.spin(box, self, "fp_cost", 1, 1000, 10,
                      callback=self._on_display_perf_line_changed)
        grid.addWidget(QtGui.QLabel("FP Cost"), 0, 0)
        grid.addWidget(sp, 0, 1)

        sp = gui.spin(box, self, "fn_cost", 1, 1000, 10,
                      callback=self._on_display_perf_line_changed)
        grid.addWidget(QtGui.QLabel("FN Cost"))
        grid.addWidget(sp, 1, 1)
        sp = gui.spin(box, self, "target_prior", 1, 99,
                      callback=self._on_display_perf_line_changed)
        sp.setSuffix("%")
        sp.addAction(QtGui.QAction("Auto", sp))
        grid.addWidget(QtGui.QLabel("Prior target class probability"))
        grid.addWidget(sp, 2, 1)

        self.plotview = pg.GraphicsView(background="w")
        self.plotview.setFrameStyle(QtGui.QFrame.StyledPanel)

        self.plot = pg.PlotItem()
        self.plot.getViewBox().setMenuEnabled(False)

        pen = QPen(self.palette().color(QtGui.QPalette.Text))

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
        """Set the input evaluation results."""
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
        self.colors = []
        self._curve_data = {}
        self._plot_curves = {}
        self._rocch = None
        self._perf_line = None

    def _initialize(self, results):
        names = getattr(results, "fitter_names", None)

        if names is None:
            names = ["#{}".format(i + 1)
                     for i in range(len(results.predicted))]

        self.colors = colorpalette.ColorPaletteGenerator(
            len(names), colorbrewer.colorSchemes["qualitative"]["Dark2"])

        self.classifier_names = names
        self.selected_classifiers = list(range(len(names)))
        for i in range(len(names)):
            listitem = self.classifiers_list_box.item(i)
            listitem.setIcon(colorpalette.ColorPixmap(self.colors[i]))

        class_var = results.data.domain.class_var
        self.target_cb.addItems(class_var.values)

    def curve_data(self, target, clf_idx):
        """Return `ROCData' for the given target and classifier."""
        if (target, clf_idx) not in self._curve_data:
            data = ROCData.from_results(self.results, clf_idx, target)
            self._curve_data[target, clf_idx] = data

        return self._curve_data[target, clf_idx]

    def plot_curves(self, target, clf_idx):
        """Return a set of functions `plot_curves` generating plot curves."""
        def generate_pens(basecolor):
            pen = QPen(basecolor, 1)
            pen.setCosmetic(True)

            shadow_pen = QPen(pen.color().lighter(160), 2.5)
            shadow_pen.setCosmetic(True)
            return pen, shadow_pen

        data = self.curve_data(target, clf_idx)

        if (target, clf_idx) not in self._plot_curves:
            pen, shadow_pen = generate_pens(self.colors[clf_idx])
            name = self.classifier_names[clf_idx]
            @once
            def merged():
                return plot_curve(
                    data.merged, pen=pen, shadow_pen=shadow_pen, name=name)
            @once
            def folds():
                return [plot_curve(fold, pen=pen, shadow_pen=shadow_pen)
                        for fold in data.folds]
            @once
            def avg_vert():
                return plot_avg_curve(data.avg_vertical, pen=pen,
                                      shadow_pen=shadow_pen, name=name)
            @once
            def avg_thres():
                return plot_avg_curve(data.avg_threshold, pen=pen,
                                      shadow_pen=shadow_pen, name=name)

            self._plot_curves[target, clf_idx] = plot_curves(
                merge=merged, folds=folds,
                avg_vertical=avg_vert, avg_threshold=avg_thres
            )

        return self._plot_curves[target, clf_idx]

    def _setup_plot(self):
        target = self.target_index
        selected = self.selected_classifiers

        curves = [self.plot_curves(target, i) for i in selected]
        selected = [self.curve_data(target, i) for i in selected]

        if self.roc_averaging == OWROCAnalysis.Merge:
            for curve in curves:
                graphics = curve.merge()
                curve = graphics.curve
                self.plot.addItem(graphics.curve_item)

                if self.display_convex_curve:
                    self.plot.addItem(graphics.hull_item)

                if self.display_def_threshold:
                    points = curve.points
                    ind = numpy.argmin(numpy.abs(points.thresholds - 0.5))
                    item = pg.TextItem(
                        text="{:.3f}".format(points.thresholds[ind]),
                    )
                    item.setPos(points.fpr[ind], points.tpr[ind])
                    self.plot.addItem(item)

            hull_curves = [curve.merged.hull for curve in selected]
            if hull_curves:
                self._rocch = convex_hull(hull_curves)
                iso_pen = QPen(QColor(Qt.black), 1)
                iso_pen.setCosmetic(True)
                self._perf_line = InfiniteLine(pen=iso_pen, antialias=True)
                self.plot.addItem(self._perf_line)

        elif self.roc_averaging == OWROCAnalysis.Vertical:
            for curve in curves:
                graphics = curve.avg_vertical()

                self.plot.addItem(graphics.curve_item)
                self.plot.addItem(graphics.confint_item)

            hull_curves = [curve.avg_vertical.hull for curve in selected]

        elif self.roc_averaging == OWROCAnalysis.Threshold:
            for curve in curves:
                graphics = curve.avg_threshold()
                self.plot.addItem(graphics.curve_item)
                self.plot.addItem(graphics.confint_item)

            hull_curves = [curve.avg_threshold.hull for curve in selected]

        elif self.roc_averaging == OWROCAnalysis.NoAveraging:
            for curve in curves:
                graphics = curve.folds()
                for fold in graphics:
                    self.plot.addItem(fold.curve_item)
                    if self.display_convex_curve:
                        self.plot.addItem(fold.hull_item)
            hull_curves = [fold.hull for curve in selected for fold in curve.folds]

        if self.display_convex_hull and hull_curves:
            hull = convex_hull(hull_curves)
            hull_pen = QPen(QColor(200, 200, 200, 100), 2)
            hull_pen.setCosmetic(True)
            item = self.plot.plot(
                hull.fpr, hull.tpr,
                pen=hull_pen,
                brush=QBrush(QColor(200, 200, 200, 50)),
                fillLevel=0)
            item.setZValue(-10000)

        pen = QPen(QColor(100, 100, 100, 100), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.plot.plot([0, 1], [0, 1], pen=pen, antialias=True)

        if self.roc_averaging == OWROCAnalysis.Merge:
            self._update_perf_line()

    def _on_target_changed(self):
        self.plot.clear()
        self._setup_plot()

    def _on_classifiers_changed(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()

    def _on_display_perf_line_changed(self):
        if self.roc_averaging == OWROCAnalysis.Merge:
            self._update_perf_line()

        if self.perf_line is not None:
            self.perf_line.setVisible(self.display_perf_line)

    def _on_display_def_threshold_changed(self):
        self._replot()

    def _replot(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()

    def _update_perf_line(self):
        if self._perf_line is None:
            return

        self._perf_line.setVisible(self.display_perf_line)
        if self.display_perf_line:
            m = roc_iso_performance_slope(
                self.fp_cost, self.fn_cost, self.target_prior / 100.0)

            hull = self._rocch
            ind = roc_iso_performance_line(m, hull)
            angle = numpy.arctan2(m, 1)  # in radians
            self._perf_line.setAngle(angle * 180 / numpy.pi)
            self._perf_line.setPos((hull.fpr[ind[0]], hull.tpr[ind[0]]))

    def onDeleteWidget(self):
        self.clear()


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
    P = numpy.sum(fold_actual == target)
    N = fold_actual.size - P

    if P == 0 or N == 0:
        # Undefined TP and FP rate
        return numpy.array([]), numpy.array([]), numpy.array([])

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


def convex_hull(curves):
    def slope(p1, p2):
        x1, y1, *_ = p1
        x2, y2, *_ = p2
        if x1 != x2:
            return  (y2 - y1) / (x2 - x1)
        else:
            return numpy.inf

    curves = [list(map(roc_point._make, zip(*curve))) for curve in curves]

    merged_points = reduce(operator.iadd, curves, [])
    merged_points = sorted(merged_points)

    if len(merged_points) == 0:
        return ROCPoints(numpy.array([]), numpy.array([]), numpy.array([]))

    if len(merged_points) <= 2:
        return ROCPoints._make(map(numpy.array, zip(*merged_points)))

    points = iter(merged_points)

    hull = deque([next(points)])

    for point in points:
        while True:
            if len(hull) < 2:
                hull.append(point)
                break
            else:
                last = hull[-1]
                if point[0] != last[0] and \
                        slope(hull[-2], last) > slope(last, point):
                    hull.append(point)
                    break
                else:
                    hull.pop()

    return ROCPoints._make(map(numpy.array, zip(*hull)))


def roc_iso_performance_line(slope, hull, tol=1e-5):
    """
    Return the indices where a line with `slope` touches the ROC convex hull.
    """
    fpr, tpr, *_ = hull

    # Compute the distance of each point to a reference iso line
    # going through point (0, 1). The point(s) with the minimum
    # distance are our result

    # y = m * x + 1
    # m * x - 1y + 1 = 0
    a, b, c = slope, -1, 1
    dist = distance_to_line(a, b, c, fpr, tpr)
    mindist = numpy.min(dist)

    return numpy.flatnonzero((dist - mindist) <= tol)


def distance_to_line(a, b, c, x0, y0):
    """
    Return the distance to a line ax + by + c = 0
    """
    assert a != 0 or b != 0
    return numpy.abs(a * x0 + b * y0 + c) / numpy.sqrt(a ** 2 + b ** 2)


def roc_iso_performance_slope(fp_cost, fn_cost, p):
    assert 0 <= p <= 1
    if fn_cost * p == 0:
        return numpy.inf
    else:
        return (fp_cost * (1. - p)) / (fn_cost * p)


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
