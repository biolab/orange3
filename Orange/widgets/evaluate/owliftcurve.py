from enum import IntEnum
from typing import NamedTuple, Dict, Tuple

import numpy as np

from AnyQt.QtWidgets import QListView, QFrame
from AnyQt.QtGui import QColor, QPen, QFont
from AnyQt.QtCore import Qt

import pyqtgraph as pg

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog
from orangewidget.widget import Msg

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.contexthandlers import \
    EvaluationResultsContextHandler
from Orange.widgets.evaluate.utils import check_results_adequacy
from Orange.widgets.utils import colorpalettes
from Orange.widgets.evaluate.owrocanalysis import convex_hull
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.customizableplot import Updater, \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import GraphicsView, PlotItem
from Orange.widgets.widget import Input
from Orange.widgets import report


CurveData = NamedTuple(
    "CurveData",
    [("contacted", np.ndarray),    # classified as positive
     ("respondents", np.ndarray),  # true positive rate
     ("thresholds", np.ndarray)]
)
CurveData.is_valid = property(lambda self: self.contacted.size > 0)


PointsAndHull = NamedTuple(
    "PointsAndHull",
    [("points", CurveData),
     ("hull", CurveData)]
)


class CurveTypes(IntEnum):
    LiftCurve, CumulativeGains = range(2)


class ParameterSetter(CommonParameterSetter):
    WIDE_LINE_LABEL = "Wide line"
    LINE_LABEL = "Thin Line"
    DEFAULT_LINE_LABEL = "Default Line"

    WIDE_LINE_WIDTH = 3
    LINE_WIDTH = 1
    DEFAULT_LINE_WIDTH = 1

    WIDE_LINE_STYLE = "Solid line"
    LINE_STYLE = "Solid line"
    DEFAULT_LINE_STYLE = "Dash line"

    def __init__(self, master):
        self.master = master
        self.wide_line_settings = {
            Updater.WIDTH_LABEL: self.WIDE_LINE_WIDTH,
            Updater.STYLE_LABEL: self.WIDE_LINE_STYLE,
        }
        self.line_settings = {
            Updater.WIDTH_LABEL: self.LINE_WIDTH,
            Updater.STYLE_LABEL: self.LINE_STYLE,
        }
        self.default_line_settings = {
            Updater.WIDTH_LABEL: self.DEFAULT_LINE_WIDTH,
            Updater.STYLE_LABEL: self.DEFAULT_LINE_STYLE,
        }
        super().__init__()

    def update_setters(self):
        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
            },
            self.ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", "")},
            },
            self.PLOT_BOX: {
                self.WIDE_LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15), self.WIDE_LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          self.WIDE_LINE_STYLE),
                },
                self.LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15), self.LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          self.LINE_STYLE),
                },
                self.DEFAULT_LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15),
                                          self.DEFAULT_LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          self.DEFAULT_LINE_STYLE),
                },
            }
        }

        def update_wide_curves(**_settings):
            self.wide_line_settings.update(**_settings)
            if self.master.display_convex_hull:
                Updater.update_lines(self.master.hull_items,
                                     **self.wide_line_settings)
            else:
                Updater.update_lines(self.master.curve_items,
                                     **self.wide_line_settings)

        def update_thin_curves(**_settings):
            self.line_settings.update(**_settings)
            if self.master.display_convex_hull:
                Updater.update_lines(self.master.curve_items,
                                     **self.line_settings)

        def update_default_line(**_settings):
            self.default_line_settings.update(**_settings)
            Updater.update_lines(self.default_line_items,
                                 **self.default_line_settings)

        self._setters[self.PLOT_BOX] = {
            self.WIDE_LINE_LABEL: update_wide_curves,
            self.LINE_LABEL: update_thin_curves,
            self.DEFAULT_LINE_LABEL: update_default_line,
        }

    @property
    def title_item(self):
        return self.master.titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in self.master.axes.values()]

    @property
    def default_line_items(self):
        return [self.master.default_line_item] \
            if self.master.default_line_item else []


class OWLiftCurve(widget.OWWidget):
    name = "Lift Curve"
    description = "Construct and display a lift curve " \
                  "from the evaluation of classifiers."
    icon = "icons/LiftCurve.svg"
    priority = 1020
    keywords = ["lift", "cumulative gain"]

    class Inputs:
        evaluation_results = Input(
            "Evaluation Results", Orange.evaluation.Results)

    class Warning(widget.OWWidget.Warning):
        undefined_curves = Msg(
            "Some curves are undefined; check models and data")

    class Error(widget.OWWidget.Error):
        undefined_curves = Msg(
            "No defined curves; check models and data")

    buttons_area_orientation = None

    settingsHandler = EvaluationResultsContextHandler()
    target_index = settings.ContextSetting(0)
    selected_classifiers = settings.ContextSetting([])

    display_convex_hull = settings.Setting(True)
    curve_type = settings.Setting(CurveTypes.LiftCurve)
    visual_settings = settings.Setting({}, schema_only=True)

    graph_name = "plot"

    YLabels = ("Lift", "TP Rate")

    def __init__(self):
        super().__init__()

        self.results = None
        self.classifier_names = []
        self.colors = []
        self._points_hull: Dict[Tuple[int, int], PointsAndHull] = {}

        box = gui.vBox(self.controlArea, box="Curve")
        self.target_cb = gui.comboBox(
            box, self, "target_index",
            label="Target: ", orientation=Qt.Horizontal,
            callback=self._on_target_changed,
            contentsLength=8, searchable=True
        )
        gui.radioButtons(
            box, self, "curve_type", ("Lift Curve", "Cumulative Gains"),
            callback=self._on_curve_type_changed
        )

        self.classifiers_list_box = gui.listBox(
            self.controlArea, self, "selected_classifiers", "classifier_names",
            box="Models",
            selectionMode=QListView.MultiSelection,
            callback=self._on_classifiers_changed
        )
        self.classifiers_list_box.setMaximumHeight(100)

        gui.checkBox(self.controlArea, self, "display_convex_hull",
                     "Show convex hull", box="Settings", callback=self._replot)

        gui.rubber(self.controlArea)

        self.plotview = GraphicsView()
        self.plotview.setFrameStyle(QFrame.StyledPanel)
        self.plot = PlotItem(enableMenu=False)
        self.plot.parameter_setter = ParameterSetter(self.plot)
        self.plot.curve_items = []
        self.plot.hull_items = []
        self.plot.default_line_item = None
        self.plot.display_convex_hull = self.display_convex_hull
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        for pos, label in (("bottom", "P Rate"), ("left", "")):
            axis = self.plot.getAxis(pos)
            axis.setTickFont(tickfont)
            axis.setLabel(label)
        self._set_left_label()

        self.plot.showGrid(True, True, alpha=0.1)

        self.plotview.setCentralItem(self.plot)
        self.mainArea.layout().addWidget(self.plotview)

        VisualSettingsDialog(self, self.plot.parameter_setter.initial_settings)

    @Inputs.evaluation_results
    def set_results(self, results):
        self.closeContext()
        self.clear()
        self.results = check_results_adequacy(results, self.Error)
        if self.results is not None:
            self._initialize(results)
            self.openContext(self.results.domain.class_var,
                             self.classifier_names)
            self._setup_plot()

    def clear(self):
        self.plot.clear()
        self.plot.curve_items = []
        self.plot.hull_items = []
        self.plot.default_line_item = None
        self.plot.display_convex_hull = self.display_convex_hull
        self.Warning.clear()
        self.Error.clear()
        self.results = None
        self.target_cb.clear()
        self.classifier_names = []
        self.colors = []
        self._points_hull = {}

    def _initialize(self, results):
        n_models = len(results.predicted)

        self.classifier_names = getattr(results, "learner_names", None) \
                                or [f"#{i}" for i in range(n_models)]
        self.selected_classifiers = list(range(n_models))

        self.colors = colorpalettes.get_default_curve_colors(n_models)
        for i, color in enumerate(self.colors):
            item = self.classifiers_list_box.item(i)
            item.setIcon(colorpalettes.ColorIcon(color))

        class_values = results.data.domain.class_var.values
        self.target_cb.addItems(class_values)
        if class_values:
            self.target_index = 0

    def _replot(self):
        self.plot.clear()
        self.plot.curve_items = []
        self.plot.hull_items = []
        self.plot.default_line_item = None
        self.plot.display_convex_hull = self.display_convex_hull
        if self.results is not None:
            self._setup_plot()

    _on_target_changed = _replot
    _on_classifiers_changed = _replot

    def _on_curve_type_changed(self):
        self._set_left_label()
        self._replot()

    def _set_left_label(self):
        self.plot.getAxis("left").setLabel(self.YLabels[self.curve_type])

    def _setup_plot(self):
        self._plot_default_line()
        is_valid = [
            self._plot_curve(self.target_index, clf_idx)
            for clf_idx in self.selected_classifiers
        ]
        self.plot.autoRange()
        self._set_undefined_curves_err_warn(is_valid)

    def _plot_curve(self, target, clf_idx):
        key = (target, clf_idx)
        if key not in self._points_hull:
            self._points_hull[key] = \
                points_from_results(self.results, target, clf_idx)
        points, hull = self._points_hull[key]

        if not points.is_valid:
            return False

        param_setter = self.plot.parameter_setter
        color = self.colors[clf_idx]
        width = param_setter.line_settings[Updater.WIDTH_LABEL]
        style = param_setter.line_settings[Updater.STYLE_LABEL]
        pen = QPen(color, width, Updater.LINE_STYLES[style])
        pen.setCosmetic(True)
        width = param_setter.wide_line_settings[Updater.WIDTH_LABEL]
        style = param_setter.wide_line_settings[Updater.STYLE_LABEL]
        wide_pen = QPen(color, width, Updater.LINE_STYLES[style])
        wide_pen.setCosmetic(True)

        def _plot(points, pen):
            contacted, respondents, _ = points
            if self.curve_type == CurveTypes.LiftCurve:
                respondents = respondents / contacted
            curve = pg.PlotCurveItem(contacted, respondents, pen=pen,
                                     antialias=True)
            self.plot.addItem(curve)
            return curve

        self.plot.curve_items.append(
            _plot(points, wide_pen if not self.display_convex_hull else pen)
        )
        if self.display_convex_hull:
            self.plot.hull_items.append(_plot(hull, wide_pen))
        return True

    def _plot_default_line(self):
        param_setter = self.plot.parameter_setter
        width = param_setter.default_line_settings[Updater.WIDTH_LABEL]
        style = param_setter.default_line_settings[Updater.STYLE_LABEL]
        pen = QPen(QColor(20, 20, 20), width, Updater.LINE_STYLES[style])
        pen.setCosmetic(True)
        y0 = 1 if self.curve_type == CurveTypes.LiftCurve else 0
        curve = pg.PlotCurveItem([0, 1], [y0, 1], pen=pen, antialias=True)
        self.plot.addItem(curve)
        self.plot.default_line_item = curve

    def _set_undefined_curves_err_warn(self, is_valid):
        self.Error.undefined_curves.clear()
        self.Warning.undefined_curves.clear()
        if not all(is_valid):
            if any(is_valid):
                self.Warning.undefined_curves()
            else:
                self.Error.undefined_curves()

    def send_report(self):
        if self.results is None:
            return
        caption = report.list_legend(self.classifiers_list_box,
                                     self.selected_classifiers)
        self.report_items((("Target class", self.target_cb.currentText()),))
        self.report_plot()
        self.report_caption(caption)

    def set_visual_settings(self, key, value):
        self.plot.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value


def points_from_results(results, target, clf_index):
    x, y, thresholds = cumulative_gains_from_results(results, target, clf_index)
    points = CurveData(x, y, thresholds)
    hull = CurveData(*convex_hull([(x, y, thresholds)]))
    return PointsAndHull(points, hull)


def cumulative_gains_from_results(results, target, clf_idx):
    y_true = results.actual
    scores = results.probabilities[clf_idx][:, target]
    yrate, tpr, thresholds = cumulative_gains(y_true, scores, target)
    return yrate, tpr, thresholds


def cumulative_gains(y_true, y_score, target=1):
    if len(y_true) != len(y_score):
        raise ValueError("array dimensions don't match")

    if not y_true.size:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    y_true = (y_true == target)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    respondents = np.cumsum(y_true)[threshold_idxs]
    respondents = respondents / respondents[-1]
    contacted = (1 + threshold_idxs) / (1 + threshold_idxs[-1])
    return contacted, respondents, y_score[threshold_idxs]


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.evaluate.utils import results_for_preview
    WidgetPreview(OWLiftCurve).run(results_for_preview())
