from typing import Optional, Tuple, Callable, List, Dict, Union

import numpy as np
from scipy.stats import spearmanr, linregress
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel
import pyqtgraph as pg

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog, \
    KeyType, ValueType
from Orange.base import Learner
from Orange.data import Table
from Orange.data.table import DomainTransformationError
from Orange.evaluation import CrossValidation, R2, AUC, TestOnTrainingData, \
    Results
from Orange.evaluation.scoring import Score
from Orange.util import dummy_callback
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.multi_target import check_multiple_targets_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import LegendItem
from Orange.widgets.visualize.utils.customizableplot import \
    CommonParameterSetter, Updater
from Orange.widgets.visualize.utils.plotutils import PlotWidget
from Orange.widgets.widget import OWWidget, Input, Msg

N_FOLD = 7
# corr, scores_tr, intercept_tr, slope_tr,
# scores_cv, intercept_cv, slope_cv, score_name
PermutationResults = \
    Tuple[np.ndarray, List, float, float, List, float, float, str]


def _f_lin(
        intercept: float,
        slope: float,
        x: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    return intercept + slope * x


def _correlation(y: np.ndarray, y_pred: np.ndarray) -> float:
    return spearmanr(y, y_pred)[0] * 100


def _validate(
        data: Table,
        learner: Learner,
        scorer: Score
) -> Tuple[float, float]:
    res: Results = TestOnTrainingData()(data, [learner],
                                        suppresses_exceptions=False)
    res_cv: Results = CrossValidation(k=N_FOLD)(data, [learner],
                                                suppresses_exceptions=False)
    # pylint: disable=unsubscriptable-object
    return scorer(res)[0], scorer(res_cv)[0]


def permutation(
        data: Table,
        learner: Learner,
        n_perm: int = 100,
        progress_callback: Callable = dummy_callback
) -> PermutationResults:
    scorer = AUC if data.domain.has_discrete_class else R2

    score_tr, score_cv = _validate(data, learner, scorer)
    scores_tr = [score_tr]
    scores_cv = [score_cv]
    correlations = [100.0]
    progress_callback(0, "Calculating...")
    np.random.seed(0)

    data_perm = data.copy()
    for i in range(n_perm):
        progress_callback(i / n_perm)
        np.random.shuffle(data_perm.Y)
        score_tr, score_cv = _validate(data_perm, learner, scorer)
        correlations.append(_correlation(data.Y, data_perm.Y))
        scores_tr.append(score_tr)
        scores_cv.append(score_cv)

    correlations = np.abs(correlations)
    res_tr = linregress([correlations[0], np.mean(correlations[1:])],
                        [scores_tr[0], np.mean(scores_tr[1:])])
    res_cv = linregress([correlations[0], np.mean(correlations[1:])],
                        [scores_cv[0], np.mean(scores_cv[1:])])

    return (correlations, scores_tr, res_tr.intercept, res_tr.slope,
            scores_cv, res_cv.intercept, res_cv.slope, scorer.name)


def run(
        data: Table,
        learner: Learner,
        n_perm: int,
        state: TaskState
) -> PermutationResults:
    def callback(i: float, status: str = ""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            # pylint: disable=broad-exception-raised
            raise Exception

    return permutation(data, learner, n_perm, callback)


class ParameterSetter(CommonParameterSetter):
    GRID_LABEL, SHOW_GRID_LABEL = "Gridlines", "Show"
    DEFAULT_ALPHA_GRID, DEFAULT_SHOW_GRID = 80, True

    def __init__(self, master):
        self.grid_settings: Dict = None
        self.master: PermutationPlot = master
        super().__init__()

    def update_setters(self):
        self.grid_settings = {
            Updater.ALPHA_LABEL: self.DEFAULT_ALPHA_GRID,
            self.SHOW_GRID_LABEL: self.DEFAULT_SHOW_GRID,
        }

        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
                self.LEGEND_LABEL: self.FONT_SETTING,
            },
            self.PLOT_BOX: {
                self.GRID_LABEL: {
                    self.SHOW_GRID_LABEL: (None, True),
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          self.DEFAULT_ALPHA_GRID),
                },
            },
        }

        def update_grid(**settings):
            self.grid_settings.update(**settings)
            self.master.showGrid(
                x=self.grid_settings[self.SHOW_GRID_LABEL],
                y=self.grid_settings[self.SHOW_GRID_LABEL],
                alpha=self.grid_settings[Updater.ALPHA_LABEL] / 255)

        self._setters[self.PLOT_BOX] = {self.GRID_LABEL: update_grid}

    @property
    def title_item(self):
        return self.master.getPlotItem().titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in
                self.master.getPlotItem().axes.values()]

    @property
    def legend_items(self):
        return self.master.legend.items


class PermutationPlot(PlotWidget):
    def __init__(self):
        super().__init__(enableMenu=False)
        self.legend = self._create_legend()
        self.parameter_setter = ParameterSetter(self)
        self.setMouseEnabled(False, False)
        self.hideButtons()

        self.showGrid(True, True)
        text = "Correlation between original Y and permuted Y (%)"
        self.setLabel(axis="bottom", text=text)

    def _create_legend(self) -> LegendItem:
        legend = LegendItem()
        legend.setParentItem(self.getViewBox())
        legend.anchor((1, 1), (1, 1), offset=(-5, -5))
        legend.hide()
        return legend

    def set_data(
            self,
            corr: np.ndarray,
            scores_tr: List,
            intercept_tr: float,
            slope_tr: float,
            scores_cv: List,
            intercept_cv: float,
            slope_cv: float,
            score_name: str
    ):
        self.clear()
        self.setLabel(axis="left", text=score_name)

        y = 0.5 if score_name == "AUC" else 0
        line = pg.InfiniteLine(pos=(0, y), angle=0, pen=pg.mkPen("#000"))

        x = np.array([0, 100])
        pen = pg.mkPen("#000", width=2, style=Qt.DashLine)
        y_tr = _f_lin(intercept_tr, slope_tr, x)
        y_cv = _f_lin(intercept_cv, slope_cv, x)
        line_tr = pg.PlotCurveItem(x, y_tr, pen=pen)
        line_cv = pg.PlotCurveItem(x, y_cv, pen=pen)

        point_pen = pg.mkPen("#333")
        kwargs_tr = {"pen": point_pen, "symbol": "o", "brush": "#6fa255"}
        kwargs_cv = {"pen": point_pen, "symbol": "s", "brush": "#3a78b6"}

        kwargs = {"size": 12, "hoverable": True,
                  "tip": 'x: {x:.3g}\ny: {y:.3g}'.format}
        kwargs.update(kwargs_tr)
        points_tr = pg.ScatterPlotItem(corr, scores_tr, **kwargs)
        kwargs.update(kwargs_cv)
        points_cv = pg.ScatterPlotItem(corr, scores_cv, **kwargs)

        self.addItem(points_tr)
        self.addItem(points_cv)
        self.addItem(line)
        self.addItem(line_tr)
        self.addItem(line_cv)

        self.legend.clear()
        self.legend.addItem(pg.ScatterPlotItem(**kwargs_tr), "Train")
        self.legend.addItem(pg.ScatterPlotItem(**kwargs_cv), "CV")
        self.legend.show()


class OWPermutationPlot(OWWidget, ConcurrentWidgetMixin):
    name = "Permutation Plot"
    description = "Permutation analysis plotting"
    icon = "icons/PermutationPlot.svg"
    priority = 1100
    keywords = "_keywords"

    n_permutations = Setting(20)
    visual_settings = Setting({}, schema_only=True)
    graph_name = "graph.plotItem"

    class Inputs:
        data = Input("Data", Table)
        learner = Input("Learner", Learner)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")
        not_enough_data = Msg(f"At least {N_FOLD} instances are needed.")
        incompatible_learner = Msg("{}")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self._data: Optional[Table] = None
        self._learner: Optional[Learner] = None
        self._info: QLabel = None
        self.graph: PermutationPlot = None
        self.setup_gui()
        VisualSettingsDialog(
            self, self.graph.parameter_setter.initial_settings
        )

    def setup_gui(self):
        self._add_plot()
        self._add_controls()

    def _add_plot(self):
        box = gui.vBox(self.mainArea)
        self.graph = PermutationPlot()
        box.layout().addWidget(self.graph)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Settings")
        gui.spin(box, self, "n_permutations", label="Permutations:",
                 minv=1, maxv=1000, callback=self._run)
        gui.rubber(self.controlArea)

        box = gui.vBox(self.controlArea, "Info")
        self._info = gui.label(box, self, "", textFormat=Qt.RichText,
                               minimumWidth=180)
        self.__set_info(None)

    def __set_info(self, result: PermutationResults):
        html = "No data available."
        if result is not None:
            intercept_tr, slope_tr, _, intercept_cv, slope_cv = result[2: -1]
            y_tr = _f_lin(intercept_tr, slope_tr, 100)
            y_cv = _f_lin(intercept_cv, slope_cv, 100)
            html = f"""
<table width=100% align="center" style="font-size:11px">
    <tr style="background:#fefefe">
        <th style="background:transparent;padding: 2px 4px"></th>
        <th style="background:transparent;padding: 2px 4px">Corr = 0</th>
        <th style="background:transparent;padding: 2px 4px">Corr = 100</th>
    </tr>
    <tr style="background:#fefefe">
        <th style="padding: 2px 4px" align=right>Train</th>
        <td style="padding: 2px 4px" align=right>{intercept_tr:.4f}</td>
        <td style="padding: 2px 4px" align=right>{y_tr:.4f}</td>
    </tr>
    <tr style="background:#fefefe">
        <th style="padding: 2px 4px" align=right>CV</th>
        <td style="padding: 2px 4px" align=right>{intercept_cv:.4f}</td>
        <td style="padding: 2px 4px" align=right>{y_cv:.4f}</td>
    </tr>
</table>
            """
        self._info.setText(html)

    @Inputs.data
    @check_multiple_targets_input
    def set_data(self, data: Table):
        self.Error.not_enough_data.clear()
        self._data = data
        if self._data and len(self._data) < N_FOLD:
            self.Error.not_enough_data()
            self._data = None

    @Inputs.learner
    def set_learner(self, learner: Learner):
        self._learner = learner

    def handleNewSignals(self):
        self.Error.incompatible_learner.clear()
        self.Error.unknown_err.clear()
        self.Error.domain_transform_err.clear()
        self.clear()
        if self._data is None or self._learner is None:
            return

        reason = self._learner.incompatibility_reason(self._data.domain)
        if reason:
            self.Error.incompatible_learner(reason)
            return

        self._run()

    def clear(self):
        self.cancel()
        self.graph.clear()
        self.graph.setTitle()
        self.__set_info(None)

    def _run(self):
        if self._data is None or self._learner is None:
            return
        self.start(run, self._data, self._learner, self.n_permutations)

    def on_done(self, result: PermutationResults):
        self.graph.set_data(*result)
        self.__set_info(result)

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def on_partial_result(self, _):
        pass

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def send_report(self):
        if self._data is None or self._learner is None:
            return
        self.report_items("Settings", [("Permutations", self.n_permutations)])
        self.report_raw("Info", self._info.text())
        self.report_name("Plot")
        self.report_plot()

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        # pylint: disable=unsupported-assignment-operation
        self.visual_settings[key] = value


if __name__ == "__main__":
    from Orange.classification import LogisticRegressionLearner

    WidgetPreview(OWPermutationPlot).run(
        set_data=Table("iris"), set_learner=LogisticRegressionLearner())
