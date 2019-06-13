from collections import namedtuple
from functools import partial

import numpy as np

from AnyQt.QtCore import Qt, QSize
from AnyQt.QtWidgets import QListWidget, QSizePolicy

import pyqtgraph as pg

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.utils import \
    check_results_adequacy, results_for_preview
from Orange.widgets.utils import colorpalette, colorbrewer
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input
from Orange.widgets import report

metric_definition = namedtuple(
    "metric_definition",
    ("name", "function", "short_names", "explanation"))


class OWCalibrationPlot(widget.OWWidget):
    name = "Calibration Plot"
    description = "Calibration plot based on evaluation of classifiers."
    icon = "icons/CalibrationPlot.svg"
    priority = 1030
    keywords = []

    class Inputs:
        evaluation_results = Input("Evaluation Results", Orange.evaluation.Results)

    class Warning(widget.OWWidget.Warning):
        empty_input = widget.Msg(
            "Empty result on input. Nothing to display.")

    target_index = settings.Setting(0)
    selected_classifiers = settings.Setting([])
    score = settings.Setting(0)
    fold_curves = settings.Setting(False)
    display_rug = settings.Setting(True)
    threshold = settings.Setting(0.5)

    graph_name = "plot"

    def __init__(self):
        super().__init__()

        self.results = None
        self.scores = None
        self.classifier_names = []
        self.colors = []

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
            callback=self._replot)

        box = gui.vBox(self.controlArea, "Metrics")
        combo = gui.comboBox(
            box, self, "score", items=(metric.name for metric in self.Metrics),
            callback=self.score_changed)

        self.explanation = gui.widgetLabel(
            box, wordWrap=True, fixedWidth=combo.sizeHint().width())
        self.explanation.setContentsMargins(8, 8, 0, 0)
        font = self.explanation.font()
        font.setPointSizeF(0.85 * font.pointSizeF())
        self.explanation.setFont(font)

        box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(box)

        self.plotview = pg.GraphicsView(background="w")
        self.plot = pg.PlotItem(enableMenu=False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()

        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=0.0))

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
        explanation = self.Metrics[self.score].explanation
        if explanation:
            self.explanation.setText(explanation)
            self.explanation.show()
        else:
            self.explanation.hide()

        axis = self.plot.getAxis("bottom")
        axis.setLabel("Predicted probability" if self.score == 0
                      else "Threshold probability to classify as positive")

        axis = self.plot.getAxis("left")
        axis.setLabel(self.Metrics[self.score].name)

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

    @staticmethod
    def plot_metrics(ytrue, probs, metrics, pen_args):
        sortind = np.argsort(probs)
        probs = probs[sortind]
        ytrue = ytrue[sortind]
        fn = np.cumsum(ytrue)
        return probs, metrics(ytrue, probs, fn, pen_args)

    def _rug(self, ytrue, probs, _fn, pen_args):
        color = pen_args["pen"].color()
        rh = 0.025
        rug_x = np.c_[probs, probs]
        rug_x_true = rug_x[ytrue].ravel()
        rug_x_false = rug_x[~ytrue].ravel()

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

    def _prob_curve(self, ytrue, probs, _fn, pen_args):
        if not probs.size:
            return

        xmin, xmax = probs.min(), probs.max()
        x = np.linspace(xmin, xmax, 100)
        if xmax != xmin:
            f = gaussian_smoother(probs, ytrue, sigma=0.15 * (xmax - xmin))
            y = f(x)
        else:
            y = np.full(100, xmax)

        self.plot.plot(x, y, symbol="+", symbolSize=4, **pen_args)
        self.plot.plot([0, 1], [0, 1], antialias=True)

    # For the following methods, at point x=i, we will have i negatives,
    # fn[i] is the number of false negatives at that point, hence
    # tn = i - fn[i]
    # tp = real_pos - fn[i]
    # fp = real_neg + tn = real_neg - (i - fn[i])

    def _ca_curve(self, ytrue, probs, fn, pen_args):
        # CA = (tn + tp) / n = ((i - fn[i]) + (real_pos - fn[i])) / n
        n = len(probs)
        real_pos = np.sum(ytrue)
        ca = (real_pos + np.arange(n) - 2 * fn) / n
        self.plot.plot(probs, ca, **pen_args)

    def _sens_spec_curve(self, ytrue, probs, fn, pen_args):
        # sens = tp / p = (real_pos - fn[i]) / real_pos
        # spec = tn / n = (i - fn[i]) / real_neg
        n = len(probs)
        real_pos = np.sum(ytrue)
        real_neg = n - real_pos
        sens = 1 - fn / real_pos
        spec = (np.arange(1, n + 1) - fn) / real_neg
        self.plot.plot(probs, sens, **pen_args)
        self.plot.plot(probs, spec, **pen_args)
        return sens, spec

    def _pr_curve(self, ytrue, probs, fn, pen_args):
        # precision = tp / pred_pos = (real_pos - fn[i]) / (n - i)
        # recall = tp / p = (real_pos - fn[i]) / real_pos
        n = len(probs)
        real_pos = np.sum(ytrue)
        fn = fn[:-1]  # prevent falling to zero at the end
        prec = (real_pos - fn) / (np.arange(n, 1, -1))
        recall = 1 - fn / real_pos
        self.plot.plot(probs[:-1], prec, **pen_args)
        self.plot.plot(probs[:-1], recall, **pen_args)
        return prec, recall

    def _ppv_npv_curve(self, ytrue, probs, fn, pen_args):
        # ppv = tp / pred_pos = (real_pos - fn[i]) / (n - i)
        # npv = tn / pred_neg = (i - fn[i]) / i
        n = len(probs)
        real_pos = np.sum(ytrue)
        fn = fn[:-1]  # prevent falling to zero at the end
        ppv = (real_pos - fn) / (np.arange(n, 1, -1))
        npv = 1 - fn / np.arange(1, n)
        self.plot.plot(probs[:-1], ppv, **pen_args)
        self.plot.plot(probs[:-1], npv, **pen_args)
        return ppv, npv

    def _setup_plot(self):
        target = self.target_index
        results = self.results
        metrics = partial(self.Metrics[self.score].function, self)
        plot_folds = self.fold_curves and results.folds is not None
        self.scores = []

        ytrue = results.actual == target
        for clsf in self.selected_classifiers:
            probs = results.probabilities[clsf, :, target]
            color = self.colors[clsf]
            pen_args = dict(
                pen=pg.mkPen(color, width=1),
                shadowPen=pg.mkPen(color.lighter(160),
                                   width=3 + 5 * plot_folds),
                antiAlias=True)
            self.scores.append(
                (self.classifier_names[clsf],
                 self.plot_metrics(ytrue, probs, metrics, pen_args)))

            if self.display_rug:
                self.plot_metrics(ytrue, probs, self._rug, pen_args)

            if plot_folds:
                pen_args = dict(
                    pen=pg.mkPen(color, width=1, style=Qt.DashLine),
                    antiAlias=True)
                for fold in range(len(results.folds)):
                    fold_results = results.get_fold(fold)
                    fold_ytrue = fold_results.actual == target
                    fold_probs = fold_results.probabilities[clsf, :, target]
                    self.plot_metrics(fold_ytrue, fold_probs, metrics, pen_args)

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
            short_names = self.Metrics[self.score].short_names
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
        ...

    def send_report(self):
        if self.results is None:
            return
        caption = report.list_legend(self.classifiers_list_box,
                                     self.selected_classifiers)
        self.report_items((("Target class", self.target_cb.currentText()),))
        self.report_plot()
        self.report_caption(caption)

    Metrics = [metric_definition(*args) for args in (
        ("Actual probability", _prob_curve, (), ""),
        ("Classification accuracy", _ca_curve, (), ""),
        ("Sensitivity and specificity", _sens_spec_curve, ("sens", "spec"),
         "<p><b>Sensitivity</b> (falling) is the proportion of correctly "
         "detected positive instances (TP / P).</p>"
         "<p><b>Specificity</b> (rising) is the proportion of detected "
         "negative instances (TP / N).</p>"),
        ("Precision and recall", _pr_curve, ("prec", "recall"),
         "<p><b>Precision</b> (rising) is the fraction of retrieved instances "
         "that are relevant, TP / (TP + FP).</p>"
         "<p><b>Recall</b> (falling) is the proportion of discovered relevant "
         "instances, TP / P.</p>"),
        ("Pos and neg predictive value", _ppv_npv_curve, ("PPV", "TPV"),
         "<p><b>Positive predictive value</b> (rising) is the proportion of "
         "correct positives, TP / (TP  + FP).</p>"
         "<p><b>Negative predictive value</b> is the proportion of correct "
         "negatives, TN / (TN + FN).</p>"),
    )]


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
