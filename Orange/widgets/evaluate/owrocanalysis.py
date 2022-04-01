import operator
from functools import reduce, wraps
from collections import namedtuple, deque, OrderedDict

import numpy as np
import sklearn.metrics as skl_metrics

from AnyQt.QtWidgets import QListView, QLabel, QGridLayout, QFrame, QAction, \
    QToolTip
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QPalette, QFont, \
    QCursor, QFontMetrics
from AnyQt.QtCore import Qt, QSize
import pyqtgraph as pg

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.contexthandlers import \
    EvaluationResultsContextHandler
from Orange.widgets.evaluate.utils import check_results_adequacy
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.plotutils import GraphicsView, PlotItem
from Orange.widgets.widget import Input
from Orange.widgets import report

from Orange.widgets.evaluate.utils import results_for_preview
from Orange.evaluation.testing import Results


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


def roc_data_from_results(results, clf_index, target):
    """
    Compute ROC Curve(s) from evaluation results.

    :param Orange.evaluation.Results results:
        Evaluation results.
    :param int clf_index:
        Learner index in the `results`.
    :param int target:
        Target class index (i.e. positive class).
    :rval ROCData:
        A instance holding the computed curves.
    """
    merged = roc_curve_for_fold(results, ..., clf_index, target)
    merged_curve = ROCCurve(ROCPoints(*merged),
                            ROCPoints(*roc_curve_convex_hull(merged)))

    folds = results.folds if results.folds is not None else [...]
    fold_curves = []
    for fold in folds:
        points = roc_curve_for_fold(results, fold, clf_index, target)
        hull = roc_curve_convex_hull(points)
        c = ROCCurve(ROCPoints(*points), ROCPoints(*hull))
        fold_curves.append(c)

    curves = [fold.points for fold in fold_curves
              if fold.is_valid]

    if curves:
        fpr, tpr, std = roc_curve_vertical_average(curves)

        thresh = np.zeros_like(fpr) * np.nan
        hull = roc_curve_convex_hull((fpr, tpr, thresh))
        v_avg = ROCAveragedVert(
            ROCPoints(fpr, tpr, thresh),
            ROCPoints(*hull),
            std
        )
    else:
        # return an invalid vertical averaged ROC
        v_avg = ROCAveragedVert(
            ROCPoints(np.array([]), np.array([]), np.array([])),
            ROCPoints(np.array([]), np.array([]), np.array([])),
            np.array([])
        )

    if curves:
        all_thresh = np.hstack([t for _, _, t in curves])
        all_thresh = np.clip(all_thresh, 0.0 - 1e-10, 1.0 + 1e-10)
        all_thresh = np.unique(all_thresh)[::-1]
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
    else:
        # return an invalid threshold averaged ROC
        t_avg = ROCAveragedThresh(
            ROCPoints(np.array([]), np.array([]), np.array([])),
            ROCPoints(np.array([]), np.array([]), np.array([])),
            np.array([]),
            np.array([])
        )
    return ROCData(merged_curve, fold_curves, v_avg, t_avg)

ROCData.from_results = staticmethod(roc_data_from_results)

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
    def extend_to_origin(points):
        "Extend ROCPoints to include coordinate origin if not already present"
        if points.tpr.size and (points.tpr[0] > 0 or points.fpr[0] > 0):
            points = ROCPoints(
                np.r_[0, points.fpr], np.r_[0, points.tpr],
                np.r_[points.thresholds[0] + 1, points.thresholds]
            )
        return points

    points = extend_to_origin(curve.points)
    item = pg.PlotCurveItem(
        points.fpr, points.tpr, pen=pen, shadowPen=shadow_pen,
        name=name, antialias=True
    )
    sp = pg.ScatterPlotItem(
        curve.points.fpr, curve.points.tpr, symbol=symbol,
        size=symbol_size, pen=shadow_pen,
        name=name
    )
    sp.setParentItem(item)

    hull = extend_to_origin(curve.hull)

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


PlotCurves = namedtuple(
    "PlotCurves",
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

    def paint(self, p, *args):
        if self.antialias:
            p.setRenderHint(QPainter.Antialiasing, True)
        super().paint(p, *args)


class OWROCAnalysis(widget.OWWidget):
    name = "ROC Analysis"
    description = "Display the Receiver Operating Characteristics curve " \
                  "based on the evaluation of classifiers."
    icon = "icons/ROCAnalysis.svg"
    priority = 1010
    keywords = []

    class Inputs:
        evaluation_results = Input("Evaluation Results", Orange.evaluation.Results)

    buttons_area_orientation = None
    settingsHandler = EvaluationResultsContextHandler()
    target_index = settings.ContextSetting(0)
    selected_classifiers = settings.ContextSetting([])

    display_perf_line = settings.Setting(True)
    display_def_threshold = settings.Setting(True)

    fp_cost = settings.Setting(500)
    fn_cost = settings.Setting(500)
    target_prior = settings.Setting(50.0, schema_only=True)

    #: ROC Averaging Types
    Merge, Vertical, Threshold, NoAveraging = 0, 1, 2, 3
    roc_averaging = settings.Setting(Merge)

    display_convex_hull = settings.Setting(False)
    display_convex_curve = settings.Setting(False)

    graph_name = "plot"

    def __init__(self):
        super().__init__()

        self.results = None
        self.classifier_names = []
        self.perf_line = None
        self.colors = []
        self._curve_data = {}
        self._plot_curves = {}
        self._rocch = None
        self._perf_line = None
        self._tooltip_cache = None

        box = gui.vBox(self.controlArea, "Plot")
        self.target_cb = gui.comboBox(
            box, self, "target_index",
            label="Target", orientation=Qt.Horizontal,
            callback=self._on_target_changed,
            contentsLength=8, searchable=True)

        gui.widgetLabel(box, "Classifiers")
        line_height = 4 * QFontMetrics(self.font()).lineSpacing()
        self.classifiers_list_box = gui.listBox(
            box, self, "selected_classifiers", "classifier_names",
            selectionMode=QListView.MultiSelection,
            callback=self._on_classifiers_changed,
            sizeHint=QSize(0, line_height))

        abox = gui.vBox(self.controlArea, "Curves")
        gui.comboBox(abox, self, "roc_averaging",
                     items=["Merge Predictions from Folds", "Mean TP Rate",
                            "Mean TP and FP at Threshold", "Show Individual Curves"],
                     callback=self._replot)

        gui.checkBox(abox, self, "display_convex_curve",
                     "Show convex ROC curves", callback=self._replot)
        gui.checkBox(abox, self, "display_convex_hull",
                     "Show ROC convex hull", callback=self._replot)

        box = gui.vBox(self.controlArea, "Analysis")

        gui.checkBox(box, self, "display_def_threshold",
                     "Default threshold (0.5) point",
                     callback=self._on_display_def_threshold_changed)

        gui.checkBox(box, self, "display_perf_line", "Show performance line",
                     callback=self._on_display_perf_line_changed)
        grid = QGridLayout()
        gui.indentedBox(box, orientation=grid)

        sp = gui.spin(box, self, "fp_cost", 1, 1000, 10,
                      alignment=Qt.AlignRight,
                      callback=self._on_display_perf_line_changed)
        grid.addWidget(QLabel("FP Cost:"), 0, 0)
        grid.addWidget(sp, 0, 1)

        sp = gui.spin(box, self, "fn_cost", 1, 1000, 10,
                      alignment=Qt.AlignRight,
                      callback=self._on_display_perf_line_changed)
        grid.addWidget(QLabel("FN Cost:"))
        grid.addWidget(sp, 1, 1)
        self.target_prior_sp = gui.spin(box, self, "target_prior", 1, 99,
                                        alignment=Qt.AlignRight,
                                        spinType=float,
                                        callback=self._on_target_prior_changed)
        self.target_prior_sp.setSuffix(" %")
        self.target_prior_sp.addAction(QAction("Auto", sp))
        grid.addWidget(QLabel("Prior probability:"))
        grid.addWidget(self.target_prior_sp, 2, 1)

        self.plotview = GraphicsView(background=None)
        self.plotview.setFrameStyle(QFrame.StyledPanel)
        self.plotview.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.plot = PlotItem(enableMenu=False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setTickFont(tickfont)
        axis.setLabel("FP Rate (1-Specificity)")
        axis.setGrid(16)

        axis = self.plot.getAxis("left")
        axis.setTickFont(tickfont)
        axis.setLabel("TP Rate (Sensitivity)")
        axis.setGrid(16)

        self.plot.showGrid(True, True, alpha=0.2)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0.05)

        self.plotview.setCentralItem(self.plot)
        self.mainArea.layout().addWidget(self.plotview)

    @Inputs.evaluation_results
    def set_results(self, results):
        """Set the input evaluation results."""
        self.closeContext()
        self.clear()
        self.results = check_results_adequacy(results, self.Error)
        if self.results is not None:
            self._initialize(self.results)
            self.openContext(self.results.domain.class_var,
                             self.classifier_names)
            self._setup_plot()
        else:
            self.warning()

    def clear(self):
        """Clear the widget state."""
        self.results = None
        self.plot.clear()
        self.classifier_names = []
        self.selected_classifiers = []
        self.target_cb.clear()
        self.colors = []
        self._curve_data = {}
        self._plot_curves = {}
        self._rocch = None
        self._perf_line = None
        self._tooltip_cache = None

    def _initialize(self, results):
        names = getattr(results, "learner_names", None)

        if names is None:
            names = ["#{}".format(i + 1)
                     for i in range(len(results.predicted))]

        self.colors = colorpalettes.get_default_curve_colors(len(names))

        self.classifier_names = names
        self.selected_classifiers = list(range(len(names)))
        for i in range(len(names)):
            listitem = self.classifiers_list_box.item(i)
            listitem.setIcon(colorpalettes.ColorIcon(self.colors[i]))

        class_var = results.data.domain.class_var
        self.target_cb.addItems(class_var.values)
        self.target_index = 0
        self._set_target_prior()

    def _set_target_prior(self):
        """
        This function sets the initial target class probability prior value
        based on the input data.
        """
        if self.results.data:
            # here we can use target_index directly since values in the
            # dropdown are sorted in same order than values in the table
            target_values_cnt = np.count_nonzero(
                self.results.data.Y == self.target_index)
            count_all = np.count_nonzero(~np.isnan(self.results.data.Y))
            self.target_prior = np.round(target_values_cnt / count_all * 100)

            # set the spin text to gray color when set automatically
            self.target_prior_sp.setStyleSheet("color: gray;")

    def curve_data(self, target, clf_idx):
        """Return `ROCData' for the given target and classifier."""
        if (target, clf_idx) not in self._curve_data:
            # pylint: disable=no-member
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

            self._plot_curves[target, clf_idx] = PlotCurves(
                merge=merged, folds=folds,
                avg_vertical=avg_vert, avg_threshold=avg_thres
            )

        return self._plot_curves[target, clf_idx]

    def _setup_plot(self):
        def merge_averaging():
            for curve in curves:
                graphics = curve.merge()
                curve = graphics.curve
                self.plot.addItem(graphics.curve_item)

                if self.display_convex_curve:
                    self.plot.addItem(graphics.hull_item)

                if self.display_def_threshold and curve.is_valid:
                    points = curve.points
                    ind = np.argmin(np.abs(points.thresholds - 0.5))
                    item = pg.TextItem(
                        text="{:.3f}".format(points.thresholds[ind]),
                        color=foreground
                    )
                    item.setPos(points.fpr[ind], points.tpr[ind])
                    self.plot.addItem(item)

            hull_curves = [curve.merged.hull for curve in selected]
            if hull_curves:
                self._rocch = convex_hull(hull_curves)
                iso_pen = QPen(foreground, 1.0)
                iso_pen.setCosmetic(True)
                self._perf_line = InfiniteLine(pen=iso_pen, antialias=True)
                self.plot.addItem(self._perf_line)
            return hull_curves

        def vertical_averaging():
            for curve in curves:
                graphics = curve.avg_vertical()

                self.plot.addItem(graphics.curve_item)
                self.plot.addItem(graphics.confint_item)
            return [curve.avg_vertical.hull for curve in selected]

        def threshold_averaging():
            for curve in curves:
                graphics = curve.avg_threshold()
                self.plot.addItem(graphics.curve_item)
                self.plot.addItem(graphics.confint_item)
            return [curve.avg_threshold.hull for curve in selected]

        def no_averaging():
            for curve in curves:
                graphics = curve.folds()
                for fold in graphics:
                    self.plot.addItem(fold.curve_item)
                    if self.display_convex_curve:
                        self.plot.addItem(fold.hull_item)
            return [fold.hull for curve in selected for fold in curve.folds]

        averagings = {
            OWROCAnalysis.Merge: merge_averaging,
            OWROCAnalysis.Vertical: vertical_averaging,
            OWROCAnalysis.Threshold: threshold_averaging,
            OWROCAnalysis.NoAveraging: no_averaging
        }
        foreground = self.plotview.scene().palette().color(QPalette.Text)
        target = self.target_index
        selected = self.selected_classifiers

        curves = [self.plot_curves(target, i) for i in selected]
        selected = [self.curve_data(target, i) for i in selected]
        hull_curves = averagings[self.roc_averaging]()

        if self.display_convex_hull and hull_curves:
            hull = convex_hull(hull_curves)
            hull_color = QColor(foreground)
            hull_color.setAlpha(100)
            hull_pen = QPen(hull_color, 2)
            hull_pen.setCosmetic(True)
            hull_color.setAlpha(50)
            item = self.plot.plot(
                hull.fpr, hull.tpr,
                pen=hull_pen,
                brush=QBrush(hull_color),
                fillLevel=0)
            item.setZValue(-10000)
        line_color = self.palette().color(QPalette.Disabled, QPalette.Text)
        pen = QPen(QColor(*line_color.getRgb()[:3], 200), 1.0, Qt.DashLine)
        pen.setCosmetic(True)
        self.plot.plot([0, 1], [0, 1], pen=pen, antialias=True)

        if self.roc_averaging == OWROCAnalysis.Merge:
            self._update_perf_line()

        self._update_axes_ticks()

        warning = ""
        if not all(c.is_valid for c in hull_curves):
            if any(c.is_valid for c in hull_curves):
                warning = "Some ROC curves are undefined"
            else:
                warning = "All ROC curves are undefined"
        self.warning(warning)

    def _update_axes_ticks(self):
        def enumticks(a):
            a = np.unique(a)
            if len(a) > 15:
                return None
            return [[(x, f"{x:.2f}") for x in a[::-1]]]

        axis_bottom = self.plot.getAxis("bottom")
        axis_left = self.plot.getAxis("left")

        if not self.selected_classifiers or len(self.selected_classifiers) > 1 \
                or self.roc_averaging != OWROCAnalysis.Merge:
            axis_bottom.setTicks(None)
            axis_left.setTicks(None)
        else:
            data = self.curve_data(self.target_index, self.selected_classifiers[0])
            points = data.merged.points
            axis_bottom.setTicks(enumticks(points.fpr))
            axis_left.setTicks(enumticks(points.tpr))

    def _on_mouse_moved(self, pos):
        target = self.target_index
        selected = self.selected_classifiers
        curves = [(clf_idx, self.plot_curves(target, clf_idx))
                  for clf_idx in selected]  # type: List[Tuple[int, PlotCurves]]
        valid_thresh, valid_clf = [], []
        pt, ave_mode = None, self.roc_averaging

        for clf_idx, crv in curves:
            if self.roc_averaging == OWROCAnalysis.Merge:
                curve = crv.merge()
            elif self.roc_averaging == OWROCAnalysis.Vertical:
                curve = crv.avg_vertical()
            elif self.roc_averaging == OWROCAnalysis.Threshold:
                curve = crv.avg_threshold()
            else:
                # currently not implemented for 'Show Individual Curves'
                return

            sp = curve.curve_item.childItems()[0]  # type: pg.ScatterPlotItem
            act_pos = sp.mapFromScene(pos)
            pts = list(sp.pointsAt(act_pos))

            if pts:
                mouse_pt = pts[0].pos()
                if self._tooltip_cache:
                    cache_pt, cache_thresh, cache_clf, cache_ave = self._tooltip_cache
                    curr_thresh, curr_clf = [], []
                    if np.linalg.norm(mouse_pt - cache_pt) < 10e-6 \
                            and cache_ave == self.roc_averaging:
                        mask = np.equal(cache_clf, clf_idx)
                        curr_thresh = np.compress(mask, cache_thresh).tolist()
                        curr_clf = np.compress(mask, cache_clf).tolist()
                    else:
                        QToolTip.showText(QCursor.pos(), "")
                        self._tooltip_cache = None

                    if curr_thresh:
                        valid_thresh.append(*curr_thresh)
                        valid_clf.append(*curr_clf)
                        pt = cache_pt
                        continue

                curve_pts = curve.curve.points
                roc_points = np.column_stack((curve_pts.fpr, curve_pts.tpr))
                diff = np.subtract(roc_points, mouse_pt)
                # Find closest point on curve and save the corresponding threshold
                idx_closest = np.argmin(np.linalg.norm(diff, axis=1))

                thresh = curve_pts.thresholds[idx_closest]
                if not np.isnan(thresh):
                    valid_thresh.append(thresh)
                    valid_clf.append(clf_idx)
                    pt = [curve_pts.fpr[idx_closest], curve_pts.tpr[idx_closest]]

        if valid_thresh:
            clf_names = self.classifier_names
            msg = "Thresholds:\n" + "\n".join(["({:s}) {:.3f}".format(clf_names[i], thresh)
                                               for i, thresh in zip(valid_clf, valid_thresh)])
            QToolTip.showText(QCursor.pos(), msg)
            self._tooltip_cache = (pt, valid_thresh, valid_clf, ave_mode)

    def _on_target_changed(self):
        self.plot.clear()
        self._set_target_prior()
        self._setup_plot()

    def _on_classifiers_changed(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()

    def _on_target_prior_changed(self):
        self.target_prior_sp.setStyleSheet("color: black;")
        self._on_display_perf_line_changed()

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
            if hull.is_valid:
                ind = roc_iso_performance_line(m, hull)
                angle = np.arctan2(m, 1)  # in radians
                self._perf_line.setAngle(angle * 180 / np.pi)
                self._perf_line.setPos((hull.fpr[ind[0]], hull.tpr[ind[0]]))
            else:
                self._perf_line.setVisible(False)

    def onDeleteWidget(self):
        self.clear()

    def send_report(self):
        if self.results is None:
            return
        items = OrderedDict()
        items["Target class"] = self.target_cb.currentText()
        if self.display_perf_line:
            items["Costs"] = \
                "FP = {}, FN = {}".format(self.fp_cost, self.fn_cost)
            items["Target probability"] = "{} %".format(self.target_prior)
        caption = report.list_legend(self.classifiers_list_box,
                                     self.selected_classifiers)
        self.report_items(items)
        self.report_plot()
        self.report_caption(caption)


def interp(x, xp, fp, left=None, right=None):
    """
    Like numpy.interp except for handling of running sequences of
    same values in `xp`.
    """
    x = np.asanyarray(x)
    xp = np.asanyarray(xp)
    fp = np.asanyarray(fp)

    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape")

    ind = np.searchsorted(xp, x, side="right")
    fx = np.zeros(len(x))

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
    P = np.sum(fold_actual == target)
    N = fold_actual.size - P

    if P == 0 or N == 0:
        # Undefined TP and FP rate
        return np.array([]), np.array([]), np.array([])

    fold_probs = res.probabilities[clf_idx][fold][:, target]
    drop_intermediate = len(fold_actual) > 20
    fpr, tpr, thresholds = skl_metrics.roc_curve(
        fold_actual, fold_probs, pos_label=target,
        drop_intermediate=drop_intermediate
    )

    # skl sets the first threshold to the highest threshold in the data + 1
    # since we deal with probabilities, we (carefully) set it to 1
    # Unrelated comparisons, thus pylint: disable=chained-comparison
    if len(thresholds) > 1 and thresholds[1] <= 1:
        thresholds[0] = 1
    return fpr, tpr, thresholds


def roc_curve_vertical_average(curves, samples=10):
    if not curves:
        raise ValueError("No curves")
    fpr_sample = np.linspace(0.0, 1.0, samples)
    tpr_samples = []
    for fpr, tpr, _ in curves:
        tpr_samples.append(interp(fpr_sample, fpr, tpr, left=0, right=1))

    tpr_samples = np.array(tpr_samples)
    return fpr_sample, tpr_samples.mean(axis=0), tpr_samples.std(axis=0)


def roc_curve_threshold_average(curves, thresh_samples):
    if not curves:
        raise ValueError("No curves")
    fpr_samples, tpr_samples = [], []
    for fpr, tpr, thresh in curves:
        ind = np.searchsorted(thresh[::-1], thresh_samples, side="left")
        ind = ind[::-1]
        ind = np.clip(ind, 0, len(thresh) - 1)
        fpr_samples.append(fpr[ind])
        tpr_samples.append(tpr[ind])

    fpr_samples = np.array(fpr_samples)
    tpr_samples = np.array(tpr_samples)

    return ((fpr_samples.mean(axis=0), fpr_samples.std(axis=0)),
            (tpr_samples.mean(axis=0), fpr_samples.std(axis=0)))


def roc_curve_thresh_avg_interp(curves, thresh_samples):
    fpr_samples, tpr_samples = [], []
    for fpr, tpr, thresh in curves:
        thresh = thresh[::-1]
        fpr = interp(thresh_samples, thresh, fpr[::-1], left=1.0, right=0.0)
        tpr = interp(thresh_samples, thresh, tpr[::-1], left=1.0, right=0.0)
        fpr_samples.append(fpr)
        tpr_samples.append(tpr)

    fpr_samples = np.array(fpr_samples)
    tpr_samples = np.array(tpr_samples)

    return ((fpr_samples.mean(axis=0), fpr_samples.std(axis=0)),
            (tpr_samples.mean(axis=0), fpr_samples.std(axis=0)))


RocPoint = namedtuple("RocPoint", ["fpr", "tpr", "threshold"])


def roc_curve_convex_hull(curve):
    def slope(p1, p2):
        x1, y1, _ = p1
        x2, y2, _ = p2
        if x1 != x2:
            return  (y2 - y1) / (x2 - x1)
        else:
            return np.inf

    fpr, _, _ = curve

    if len(fpr) <= 2:
        return curve
    points = map(RocPoint._make, zip(*curve))

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

    fpr = np.array([p.fpr for p in hull])
    tpr = np.array([p.tpr for p in hull])
    thres = np.array([p.threshold for p in hull])
    return (fpr, tpr, thres)


def convex_hull(curves):
    def slope(p1, p2):
        x1, y1, *_ = p1
        x2, y2, *_ = p2
        if x1 != x2:
            return (y2 - y1) / (x2 - x1)
        else:
            return np.inf

    curves = [list(map(RocPoint._make, zip(*curve))) for curve in curves]

    merged_points = reduce(operator.iadd, curves, [])
    merged_points = sorted(merged_points)

    if not merged_points:
        return ROCPoints(np.array([]), np.array([]), np.array([]))

    if len(merged_points) <= 2:
        return ROCPoints._make(map(np.array, zip(*merged_points)))

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

    return ROCPoints._make(map(np.array, zip(*hull)))


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
    mindist = np.min(dist)

    return np.flatnonzero((dist - mindist) <= tol)


def distance_to_line(a, b, c, x0, y0):
    """
    Return the distance to a line ax + by + c = 0
    """
    assert a != 0 or b != 0
    return np.abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)


def roc_iso_performance_slope(fp_cost, fn_cost, p):
    assert 0 <= p <= 1
    if fn_cost * p == 0:
        return np.inf
    else:
        return (fp_cost * (1. - p)) / (fn_cost * p)


def _create_results():  # pragma: no cover
    probs1 = [0.984, 0.907, 0.881, 0.865, 0.815, 0.741, 0.735, 0.635,
              0.582, 0.554, 0.413, 0.317, 0.287, 0.225, 0.216, 0.183]
    probs = np.array([[[1 - x, x] for x in probs1]])
    preds = (probs > 0.5).astype(float)
    return Results(
        data=Orange.data.Table("heart_disease")[:16],
        row_indices=np.arange(16),
        actual=np.array(list(map(int, "1100111001001000"))),
        probabilities=probs, predicted=preds
    )


if __name__ == "__main__":  # pragma: no cover
    # WidgetPreview(OWROCAnalysis).run(_create_results())
    WidgetPreview(OWROCAnalysis).run(results_for_preview())
