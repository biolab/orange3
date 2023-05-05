from enum import IntEnum
from typing import NamedTuple, Dict, Tuple, List

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

from AnyQt.QtWidgets import QListView, QFrame
from AnyQt.QtGui import QColor, QPen, QFont
from AnyQt.QtCore import Qt

import pyqtgraph as pg

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog
from orangewidget.widget import Msg

import Orange
from Orange.base import Model
from Orange.classification import ThresholdClassifier
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.contexthandlers import \
    EvaluationResultsContextHandler
from Orange.widgets.evaluate.utils import check_results_adequacy
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.customizableplot import Updater, \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import GraphicsView, PlotItem
from Orange.widgets.widget import Input, Output
from Orange.widgets import report


CurveData = NamedTuple(
    "CurveData",
    [("contacted", np.ndarray),    # classified as positive
     ("respondents", np.ndarray),  # true positive rate
     ("thresholds", np.ndarray)]
)
CurveData.is_valid = property(lambda self: self.contacted.size > 0)


class CurveTypes(IntEnum):
    LiftCurve, CumulativeGains, PrecisionRecall = range(3)  # pylint: disable=invalid-name


class ParameterSetter(CommonParameterSetter):
    WIDE_LINE_LABEL = "Line"
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
            Updater.update_lines(self.master.curve_items,
                                 **self.wide_line_settings)

        def update_default_line(**_settings):
            self.default_line_settings.update(**_settings)
            Updater.update_lines(self.default_line_items,
                                 **self.default_line_settings)

        self._setters[self.PLOT_BOX] = {
            self.WIDE_LINE_LABEL: update_wide_curves,
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
    name = "Performance Curve"
    description = "Construct and display a performance curve " \
                  "from the evaluation of classifiers."
    icon = "icons/LiftCurve.svg"
    priority = 1020
    keywords = "performance curve, lift, cumulative gain, precision, recall, curve"

    class Inputs:
        evaluation_results = Input(
            "Evaluation Results", Orange.evaluation.Results)

    class Outputs:
        calibrated_model = Output("Calibrated Model", Model)

    class Warning(widget.OWWidget.Warning):
        undefined_curves = Msg(
            "Some curves are undefined; check models and data")

    class Error(widget.OWWidget.Error):
        undefined_curves = Msg("No defined curves; check models and data")

    class Information(widget.OWWidget.Information):
        no_output = Msg("Can't output a model: {}")

    settingsHandler = EvaluationResultsContextHandler()
    target_index = settings.ContextSetting(0)
    selected_classifiers = settings.ContextSetting([])

    curve_type = settings.Setting(CurveTypes.LiftCurve)
    show_threshold = settings.Setting(True)
    show_points = settings.Setting(True)
    rate = settings.Setting(0.5)
    auto_commit = settings.Setting(True)
    visual_settings = settings.Setting({}, schema_only=True)

    graph_name = "plot"  # pg.GraphicsItem (pg.PlotItem)

    XLabels = ("P Rate", "P Rate", "Recall")
    YLabels = ("Lift", "TP Rate", "Precision")

    def __init__(self):
        super().__init__()

        self.results = None
        self.classifier_names = []
        self.colors = []
        self._points: Dict[Tuple[int, int, int], CurveData] = {}
        self.line = None
        self.tooltip = None

        box = gui.vBox(self.controlArea, box="Curve")
        self.target_cb = gui.comboBox(
            box, self, "target_index",
            label="Target: ", orientation=Qt.Horizontal,
            callback=self._on_target_changed,
            contentsLength=8, searchable=True
        )
        gui.radioButtons(
            box, self, "curve_type",
            ("Lift Curve", "Cumulative Gains", "Precision Recall"),
            callback=self._on_curve_type_changed
        )

        self.classifiers_list_box = gui.listBox(
            self.controlArea, self, "selected_classifiers", "classifier_names",
            box="Models",
            selectionMode=QListView.MultiSelection,
            callback=self._on_classifiers_changed
        )
        self.classifiers_list_box.setMaximumHeight(100)

        box = gui.vBox(self.controlArea, box="Settings")
        gui.checkBox(box, self, "show_threshold", "Show thresholds",
                     callback=self._on_show_threshold_changed)
        gui.checkBox(box, self, "show_points", "Show points",
                     callback=self._on_show_points_changed)

        gui.rubber(self.controlArea)

        box = gui.vBox(self.controlArea, box="Area under the curve")
        self._area_info = gui.label(box, self, "/", textFormat=Qt.RichText)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

        self.plotview = GraphicsView()
        self.plotview.setFrameStyle(QFrame.StyledPanel)
        self.plot = PlotItem(enableMenu=False)
        self.plot.parameter_setter = ParameterSetter(self.plot)
        self.plot.curve_items = []
        self.plot.default_line_item = None
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))
        for pos in ("bottom", "left"):
            axis = self.plot.getAxis(pos)
            axis.setTickFont(tickfont)
        self._set_axes_labels()

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
        self.commit.now()

    def clear(self):
        self.plot.clear()
        self.plot.curve_items = []
        self.plot.default_line_item = None
        self.Warning.clear()
        self.Error.clear()
        self.results = None
        self.target_cb.clear()
        self.classifier_names = []
        self.colors = []
        self._points = {}
        self._update_info([])

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
        self.plot.default_line_item = None
        if self.results is not None:
            self._setup_plot()

    def _on_target_changed(self):
        self._replot()
        self.commit.deferred()

    def _on_classifiers_changed(self):
        self._on_show_threshold_changed()
        self._replot()
        self.commit.deferred()

    def _on_curve_type_changed(self):
        self._set_axes_labels()
        self._replot()
        self.commit.deferred()

    def _on_threshold_change(self):
        self.rate = round(self.line.pos().x(), 5)
        self.line.setPos(self.rate)
        self._set_tooltip()

    def _on_show_threshold_changed(self):
        selected = len(self.selected_classifiers) > 0
        self.tooltip.setVisible(self.show_threshold and selected)

    def _on_show_points_changed(self):
        for item in self.plot.curve_items:
            item.scatter.setVisible(self.show_points)

    def _set_axes_labels(self):
        self.plot.getAxis("bottom").setLabel(
            self.XLabels[int(self.curve_type)])
        self.plot.getAxis("left").setLabel(self.YLabels[int(self.curve_type)])

    def _setup_plot(self):
        self._plot_default_line()
        is_valid = [
            self._plot_curve(self.target_index, clf_idx, self.curve_type)
            for clf_idx in self.selected_classifiers
        ]
        self._update_info(is_valid)
        self.plot.autoRange()
        if self.curve_type != CurveTypes.LiftCurve:
            self.plot.getViewBox().setYRange(0, 1)
        self._set_undefined_curves_err_warn(is_valid)

        self.line = pg.InfiniteLine(
            pos=self.rate, movable=True,
            pen=pg.mkPen(color="k", style=Qt.DashLine, width=2),
            hoverPen=pg.mkPen(color="k", style=Qt.DashLine, width=3),
            bounds=(0, 1),
        )
        self.line.setCursor(Qt.SizeHorCursor)
        self.line.sigPositionChanged.connect(self._on_threshold_change)
        self.line.sigPositionChangeFinished.connect(
            self._on_threshold_change_done)
        self.plot.addItem(self.line)

        self.tooltip = pg.TextItem(border=QColor(*(100, 100, 100, 200)),
                                   fill=(250, 250, 250, 200))
        self.tooltip.setZValue(1e9)
        self.plot.addItem(self.tooltip)
        self._set_tooltip()

        self._on_show_points_changed()
        self._on_show_threshold_changed()

    def _on_threshold_change_done(self):
        self.commit.deferred()

    def _plot_curve(self, target, clf_idx, curve_type):
        curve_type = curve_type if curve_type == CurveTypes.PrecisionRecall \
            else CurveTypes.LiftCurve
        key = (target, clf_idx, curve_type)
        if key not in self._points:
            self._points[key] = points_from_results(
                self.results, target, clf_idx, self.curve_type)
        points = self._points[key]

        if not points.is_valid:
            return False

        param_setter = self.plot.parameter_setter
        color = self.colors[clf_idx]
        width = param_setter.wide_line_settings[Updater.WIDTH_LABEL]
        style = param_setter.wide_line_settings[Updater.STYLE_LABEL]
        wide_pen = QPen(color, width, Updater.LINE_STYLES[style])
        wide_pen.setCosmetic(True)

        def tip(x, y, data):
            xlabel = self.XLabels[int(self.curve_type)]
            ylabel = self.YLabels[int(self.curve_type)]
            return f"{xlabel}: {round(x, 3)}\n" \
                   f"{ylabel}: {round(y, 3)}\n" \
                   f"Threshold: {round(data, 3)}"

        def _plot(points, pen, kwargs):
            contacted, respondents, _ = points
            if self.curve_type == CurveTypes.LiftCurve:
                respondents = respondents / contacted
            curve = pg.PlotDataItem(contacted, respondents, pen=pen,
                                    antialias=True, **kwargs)
            curve.scatter.opts["hoverable"] = True
            curve.scatter.opts["tip"] = tip
            self.plot.addItem(curve)
            bottom = pg.PlotDataItem(contacted, np.zeros(len(contacted)))
            area_color = QColor(color)
            area_color.setAlphaF(0.1)
            area_item = pg.FillBetweenItem(curve, bottom,
                                           brush=pg.mkBrush(area_color))
            self.plot.addItem(area_item)
            return curve

        light_color = QColor(color)
        light_color.setAlphaF(0.25)
        line_kwargs = {"symbol": "o", "symbolSize": 8,
                       "symbolPen": light_color, "symbolBrush": light_color,
                       "data": points.thresholds, "stepMode": "right"}

        self.plot.curve_items.append(_plot(points, wide_pen, line_kwargs))
        return True

    def _update_info(self, is_valid: List[bool]):
        self._area_info.setText("/")
        if any(is_valid):
            text = "<table>"
            for clf_idx, valid in zip(self.selected_classifiers, is_valid):
                if valid:
                    if self.curve_type == CurveTypes.PrecisionRecall:
                        curve_type = self.curve_type
                    else:
                        curve_type = CurveTypes.LiftCurve
                    key = self.target_index, clf_idx, curve_type
                    contacted, respondents, _ = self._points[key]
                    if self.curve_type == CurveTypes.LiftCurve:
                        respondents = respondents / contacted
                    area = compute_area(contacted, respondents)
                    area = f"{round(area, 3)}"
                else:
                    area = "/"
                text += \
                    f"<tr align=left>" \
                    f"<td style='color:{self.colors[clf_idx].name()}'>■</td>" \
                    f"<td>{self.classifier_names[clf_idx]}:  </td>" \
                    f"<td align=right>{area}</td>" \
                    f"</tr>"
            text += "<table>"
            self._area_info.setText(text)

    def _set_tooltip(self):
        html = ""
        if len(self.plot.curve_items) > 0:
            html = '<div style="color: #333; font-size: 12px;"' \
                   ' <span>Probability threshold(s):</span>'
            for item in self.plot.curve_items:
                threshold = self._get_threshold(item.xData, item.opts["data"])
                html += f'<div>' \
                        f'<span style="font-weight: 700; ' \
                        f'color: {item.opts["pen"].color().name()}"> — </span>' \
                        f'<span>{round(threshold, 3)}</span>' \
                        f'</div>'
            html += '</div>'

        self.tooltip.setHtml(html)

        view_box = self.plot.getViewBox()
        y_min, y_max = view_box.viewRange()[1]
        self.tooltip.setPos(self.rate, y_min + (y_max - y_min) * 0.8)
        half_width = self.tooltip.boundingRect().width() * \
            view_box.viewPixelSize()[0] / 2
        anchor = [0.5, 0]
        if half_width > self.rate - 0:
            anchor[0] = 0
        elif half_width > 1 - self.rate:
            anchor[0] = 1
        self.tooltip.setAnchor(anchor)

    def _get_threshold(self, contacted, thresholds):
        indices = np.array(thresholds).argsort()[::-1]
        diff = contacted[indices]
        value = diff[diff - self.rate >= 0][0]
        ind = np.where(np.round(contacted, 6) == np.round(value, 6))[0][-1]
        return thresholds[ind]

    def _plot_default_line(self):
        param_setter = self.plot.parameter_setter
        width = param_setter.default_line_settings[Updater.WIDTH_LABEL]
        style = param_setter.default_line_settings[Updater.STYLE_LABEL]
        pen = QPen(QColor(20, 20, 20), width, Updater.LINE_STYLES[style])
        pen.setCosmetic(True)
        if self.curve_type == CurveTypes.LiftCurve:
            y0, y1 = 1, 1
        elif self.curve_type == CurveTypes.CumulativeGains:
            y0, y1 = 0, 1
        else:
            y_true = self.results.actual
            y0 = y1 = sum(y_true == self.target_index) / len(y_true)
        curve = pg.PlotCurveItem([0, 1], [y0, y1], pen=pen, antialias=True)
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

    @gui.deferred
    def commit(self):
        self.Information.no_output.clear()
        wrapped = None
        results = self.results
        if results is not None:
            problems = [
                msg for condition, msg in (
                    (results.folds is not None and len(results.folds) > 1,
                     "each training data sample produces a different model"),
                    (results.models is None,
                     "test results do not contain stored models - try testing "
                     "on separate data or on training data"),
                    (len(self.selected_classifiers) != 1,
                     "select a single model - the widget can output only one"),
                    (len(results.domain.class_var.values) != 2,
                     "cannot calibrate non-binary classes"))
                if condition]
            if len(problems) == 1:
                self.Information.no_output(problems[0])
            elif problems:
                self.Information.no_output(
                    "".join(f"\n - {problem}" for problem in problems))
            else:
                clsf_idx = self.selected_classifiers[0]
                model = results.models[0, clsf_idx]
                item = self.plot.curve_items[0]
                threshold = self._get_threshold(item.xData, item.opts["data"])
                threshold = [1 - threshold, threshold][self.target_index]
                wrapped = ThresholdClassifier(model, threshold)

        self.Outputs.calibrated_model.send(wrapped)

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


def points_from_results(results, target, clf_index, curve_type):
    func = precision_recall_from_results \
        if curve_type == CurveTypes.PrecisionRecall \
        else cumulative_gains_from_results
    x, y, thresholds = func(results, target, clf_index)
    return CurveData(x, y, thresholds)


def precision_recall_from_results(results, target, clf_idx):
    y_true = results.actual
    classes = np.unique(results.actual)
    if len(classes) > 2:
        y_true = label_binarize(y_true, classes=sorted(classes))
        y_true = y_true[:, target]
    scores = results.probabilities[clf_idx][:, target]
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # scikit's precision_recall_curve adds a (0, 1) point,
    # so we add a corresponding threshold = 1.
    # In case the probability threshold was 1 we remove the (0, 1) point.
    if thresholds[-1] < 1:
        thresholds = np.append(thresholds, 1)
    else:
        recall = recall[:-1]
        precision = precision[:-1]

    return recall, precision, thresholds


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


def compute_area(x: np.ndarray, y: np.ndarray) -> float:
    ids = np.argsort(x)
    x = x[ids]
    y = y[ids]
    return np.dot(x[1:] - x[:-1], y[:-1])


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.evaluate.utils import results_for_preview
    WidgetPreview(OWLiftCurve).run(results_for_preview())
