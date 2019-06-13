from collections import namedtuple

import numpy as np

from AnyQt.QtCore import Qt, QSize
from AnyQt.QtWidgets import QListWidget, QSizePolicy

import pyqtgraph as pg

from Orange.classification import ModelWithThreshold
from Orange.evaluation import Results
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.utils import \
    check_results_adequacy, results_for_preview
from Orange.widgets.utils import colorpalette, colorbrewer
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, Msg
from Orange.widgets import report


class Data:
    def __init__(self, ytrue, probs):
        sortind = np.argsort(probs)
        self.probs = probs[sortind]
        self.ytrue = ytrue[sortind]
        self.fn = np.cumsum(self.ytrue)
        self.tot = len(probs)
        self.p = self.fn[-1]
        self.n = self.tot - self.p

    @property
    def tn(self):
        return np.arange(self.tot) - self.fn

    @property
    def tp(self):
        return self.p - self.fn

    @property
    def fp(self):
        return self.n - self.tn


MetricDefinition = namedtuple(
    "metric_definition",
    ("name", "function", "short_names", "explanation"))

Metrics = [MetricDefinition(*args) for args in (
    ("Actual probability",
     None,
     (),
     ""),
    ("Classification accuracy",
     lambda d: (d.probs, ((d.tp + d.tn) / d.tot,)),
     (),
     ""),
    ("F1",
     lambda d: (d.probs, (2 * d.tp / (2 * d.tp + d.fp + d.fn),)),
     (),
     ""),
    ("Sensitivity and specificity",
     lambda d: (d.probs, (d.tp / d.p, d.tn / d.n)),
     ("sens", "spec"),
     "<p><b>Sensitivity</b> (falling) is the proportion of correctly "
     "detected positive instances (TP&nbsp;/&nbsp;P).</p>"
     "<p><b>Specificity</b> (rising) is the proportion of detected "
     "negative instances (TP&nbsp;/&nbsp;N).</p>"),
    ("Precision and recall",
     lambda d: (d.probs[:-1], (d.tp[:-1] / np.arange(d.tot, 1, -1),
                               d.tp[:-1] / d.p)),
     ("prec", "recall"),
     "<p><b>Precision</b> (rising) is the fraction of retrieved instances "
     "that are relevant, TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FP).</p>"
     "<p><b>Recall</b> (falling) is the proportion of discovered relevant "
     "instances, TP&nbsp;/&nbsp;P.</p>"),
    ("Pos and neg predictive value",
     lambda d: (d.probs[:-1], (d.tp[:-1] / np.arange(d.tot, 1, -1),
                               d.tn[:-1] / np.arange(1, d.tot))),
     ("PPV", "TPV"),
     "<p><b>Positive predictive value</b> (rising) is the proportion of "
     "correct positives, TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FP).</p>"
     "<p><b>Negative predictive value</b> is the proportion of correct "
     "negatives, TN&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FN).</p>"),
    ("True and false positive rate",
     lambda d: (d.probs, (d.tp / d.p, d.fp / d.n)),
     ("TPR", "FPR"),
     "<p><b>True and false positive rate</b> are proportions of detected "
     "and omitted positive instances</p>"),
)]


class OWCalibrationPlot(widget.OWWidget):
    name = "Calibration Plot"
    description = "Calibration plot based on evaluation of classifiers."
    icon = "icons/CalibrationPlot.svg"
    priority = 1030
    keywords = []

    class Inputs:
        evaluation_results = Input("Evaluation Results", Results)

    class Outputs:
        calibrated_model = Output("Calibrated Model", ModelWithThreshold)

    class Warning(widget.OWWidget.Warning):
        empty_input = widget.Msg(
            "Empty result on input. Nothing to display.")

    class Information(widget.OWWidget.Information):
        no_out = "Can't output a model: "
        no_output_multiple_folds = Msg(
            no_out + "every training data sample produced a different model")
        no_output_no_models = Msg(
            no_out + "test results do not contain stored models;\n"
            "try testing on separate data or on training data")
        no_output_multiple_selected = Msg(
            no_out + "select a single model - the widget can output only one")


    target_index = settings.Setting(0)
    selected_classifiers = settings.Setting([])
    score = settings.Setting(0)
    fold_curves = settings.Setting(False)
    display_rug = settings.Setting(True)
    threshold = settings.Setting(0.5)
    auto_commit = settings.Setting(True)

    graph_name = "plot"

    def __init__(self):
        super().__init__()

        self.results = None
        self.scores = None
        self.classifier_names = []
        self.colors = []
        self.line = None

        box = gui.vBox(self.controlArea, box="Settings")
        self.target_cb = gui.comboBox(
            box, self, "target_index", label="Target:",
            orientation=Qt.Horizontal, callback=self._replot, contentsLength=8)
        gui.checkBox(
            box, self, "display_rug", "Show rug",
            callback=self._on_display_rug_changed)
        gui.checkBox(
            box, self, "fold_curves", "Curves for individual folds",
            callback=self._replot)

        self.classifiers_list_box = gui.listBox(
            self.controlArea, self, "selected_classifiers", "classifier_names",
            box="Classifier", selectionMode=QListWidget.ExtendedSelection,
            sizePolicy=(QSizePolicy.Preferred, QSizePolicy.Preferred),
            sizeHint=QSize(150, 40),
            callback=self._on_selection_changed)

        box = gui.vBox(self.controlArea, "Metrics")
        combo = gui.comboBox(
            box, self, "score", items=(metric.name for metric in Metrics),
            callback=self.score_changed)

        self.explanation = gui.widgetLabel(
            box, wordWrap=True, fixedWidth=combo.sizeHint().width())
        self.explanation.setContentsMargins(8, 8, 0, 0)
        font = self.explanation.font()
        font.setPointSizeF(0.85 * font.pointSizeF())
        self.explanation.setFont(font)

        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box)

        gui.auto_commit(
            self.controlArea, self, "auto_commit", "Apply", commit=self.apply)

        self.plotview = pg.GraphicsView(background="w")
        self.plot = pg.PlotItem(enableMenu=False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()

        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=0.0))
            if axis_name != "bottom":  # remove if when pyqtgraph is fixed
                axis.setStyle(stopAxisAtTick=(True, True))

        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0.05)
        self.plotview.setCentralItem(self.plot)

        self.mainArea.layout().addWidget(self.plotview)
        self._set_explanation()

    @Inputs.evaluation_results
    def set_results(self, results):
        self.clear()
        results = check_results_adequacy(results, self.Error)
        if results is not None and not results.actual.size:
            self.Warning.empty_input()
        else:
            self.Warning.empty_input.clear()
        self.results = results
        if self.results is not None:
            self._initialize(results)
            self._replot()
        self.apply()

    def clear(self):
        self.plot.clear()
        self.results = None
        self.classifier_names = []
        self.selected_classifiers = []
        self.target_cb.clear()
        self.target_index = 0
        self.colors = []

    def score_changed(self):
        self._set_explanation()
        self._replot()

    def _set_explanation(self):
        explanation = Metrics[self.score].explanation
        if explanation:
            self.explanation.setText(explanation)
            self.explanation.show()
        else:
            self.explanation.hide()

        axis = self.plot.getAxis("bottom")
        axis.setLabel("Predicted probability" if self.score == 0
                      else "Threshold probability to classify as positive")

        axis = self.plot.getAxis("left")
        axis.setLabel(Metrics[self.score].name)

    def _initialize(self, results):
        N = len(results.predicted)
        names = getattr(results, "learner_names", None)
        if names is None:
            names = ["#{}".format(i + 1) for i in range(N)]

        self.classifier_names = names
        scheme = colorbrewer.colorSchemes["qualitative"]["Dark2"]
        if N > len(scheme):
            scheme = colorpalette.DefaultRGBColors
        self.colors = colorpalette.ColorPaletteGenerator(N, scheme)

        for i in range(N):
            item = self.classifiers_list_box.item(i)
            item.setIcon(colorpalette.ColorPixmap(self.colors[i]))

        self.selected_classifiers = list(range(N))
        self.target_cb.addItems(results.data.domain.class_var.values)

    def _rug(self, data, pen_args):
        color = pen_args["pen"].color()
        rh = 0.025
        rug_x = np.c_[data.probs, data.probs]
        rug_x_true = rug_x[data.ytrue].ravel()
        rug_x_false = rug_x[~data.ytrue].ravel()

        rug_y_true = np.ones_like(rug_x_true)
        rug_y_true[1::2] = 1 - rh
        rug_y_false = np.zeros_like(rug_x_false)
        rug_y_false[1::2] = rh

        self.plot.plot(
            rug_x_false, rug_y_false,
            pen=color, connect="pairs", antialias=True)
        self.plot.plot(
            rug_x_true, rug_y_true,
            pen=color, connect="pairs", antialias=True)

    def plot_metrics(self, data, metrics, pen_args):
        if metrics is None:
            return self._prob_curve(data.ytrue, data.probs, pen_args)
        x, ys = metrics(data)
        for y in ys:
            self.plot.plot(x, y, **pen_args)
        return x, ys

    def _prob_curve(self, ytrue, probs, pen_args):
        if not probs.size:
            return None

        xmin, xmax = probs.min(), probs.max()
        x = np.linspace(xmin, xmax, 100)
        if xmax != xmin:
            f = gaussian_smoother(probs, ytrue, sigma=0.15 * (xmax - xmin))
            y = f(x)
        else:
            y = np.full(100, xmax)

        self.plot.plot(x, y, symbol="+", symbolSize=4, **pen_args)
        self.plot.plot([0, 1], [0, 1], antialias=True)
        return x, (y, )

    def _setup_plot(self):
        target = self.target_index
        results = self.results
        metrics = Metrics[self.score].function
        plot_folds = self.fold_curves and results.folds is not None
        self.scores = []

        ytrue = results.actual == target
        for clsf in self.selected_classifiers:
            probs = results.probabilities[clsf, :, target]
            color = self.colors[clsf]
            pen_args = dict(
                pen=pg.mkPen(color, width=1),
                shadowPen=pg.mkPen(color.lighter(160),
                                   width=4 + 4 * plot_folds),
                antiAlias=True)
            data = Data(ytrue, probs)
            self.scores.append(
                (self.classifier_names[clsf],
                 self.plot_metrics(data, metrics, pen_args)))

            if self.display_rug:
                self._rug(data, pen_args)

            if plot_folds:
                pen_args = dict(
                    pen=pg.mkPen(color, width=1, style=Qt.DashLine),
                    antiAlias=True)
                for fold in range(len(results.folds)):
                    fold_results = results.get_fold(fold)
                    fold_ytrue = fold_results.actual == target
                    fold_probs = fold_results.probabilities[clsf, :, target]
                    self.plot_metrics(Data(fold_ytrue, fold_probs),
                                      metrics, pen_args)

    def _replot(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()
        self.line = pg.InfiniteLine(
            pos=self.threshold, movable=True,
            pen=pg.mkPen(color="k", style=Qt.DashLine, width=2),
            hoverPen=pg.mkPen(color="k", style=Qt.DashLine, width=3),
            bounds=(0, 1),
        )
        self.line.sigPositionChanged.connect(self.threshold_change)
        self.line.sigPositionChangeFinished.connect(self.threshold_change_done)
        self.plot.addItem(self.line)
        self._update_info()

    def _on_display_rug_changed(self):
        self._replot()

    def _on_selection_changed(self):
        self._replot()
        self.apply()

    def threshold_change(self):
        self.threshold = round(self.line.pos().x(), 2)
        self.line.setPos(self.threshold)
        self._update_info()

    def _update_info(self):
        text = f"""<table>
                        <tr>
                            <th align='right'>Threshold: p=</th>
                            <td colspan='4'>{self.threshold:.2f}<br/></td>
                        </tr>"""
        if self.scores is not None:
            short_names = Metrics[self.score].short_names
            if short_names:
                text += f"""<tr>
                                <th></th>
                                {"<td></td>".join(f"<td align='right'>{n}</td>"
                                                  for n in short_names)}
                            </tr>"""
            for name, (probs, curves) in self.scores:
                ind = min(np.searchsorted(probs, self.threshold),
                          len(probs) - 1)
                text += f"<tr><th align='right'>{name}:</th>"
                text += "<td>/</td>".join(f'<td>{curve[ind]:.3f}</td>'
                                          for curve in curves)
                text += "</tr>"
            text += "<table>"
        self.info_label.setText(text)

    def threshold_change_done(self):
        self.apply()

    def apply(self):
        info = self.Information
        wrapped = None
        problems = {}
        if self.results is not None:
            problems = {
                info.no_output_multiple_folds: len(self.results.folds) > 1,
                info.no_output_no_models: self.results.models is None,
                info.no_output_multiple_selected:
                    len(self.selected_classifiers) != 1}
            if not any(problems.values()):
                model = self.results.models[0][self.selected_classifiers[0]]
                wrapped = ModelWithThreshold(model, self.threshold)

        self.Outputs.calibrated_model.send(wrapped)
        for info, shown in problems.items():
            if info.is_shown() != shown:
                info(shown=shown)

    def send_report(self):
        if self.results is None:
            return
        caption = report.list_legend(self.classifiers_list_box,
                                     self.selected_classifiers)
        self.report_items((("Target class", self.target_cb.currentText()),))
        self.report_plot()
        self.report_caption(caption)


def gaussian_smoother(x, y, sigma=1.0):
    x = np.asarray(x)
    y = np.asarray(y)

    gamma = 1. / (2 * sigma ** 2)
    a = 1. / (sigma * np.sqrt(2 * np.pi))

    if x.shape != y.shape:
        raise ValueError

    def smoother(xs):
        W = a * np.exp(-gamma * ((xs - x) ** 2))
        return np.average(y, weights=W)

    return np.vectorize(smoother, otypes=[np.float])


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCalibrationPlot).run(results_for_preview())
