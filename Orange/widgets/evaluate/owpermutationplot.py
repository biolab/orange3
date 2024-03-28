from typing import Optional, Tuple, Callable, List, Dict

import numpy as np
from scipy.stats import spearmanr, linregress
from AnyQt.QtCore import Qt
import pyqtgraph as pg

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog, \
    KeyType, ValueType
from Orange.base import Learner
from Orange.data import Table
from Orange.data.table import DomainTransformationError
from Orange.evaluation import CrossValidation, R2, TestOnTrainingData, Results
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
PermutationResults = Tuple[np.ndarray, List, float, float, List, float, float]


def _correlation(y: np.ndarray, y_pred: np.ndarray) -> float:
    return spearmanr(y, y_pred)[0] * 100


def _validate(data: Table, learner: Learner) -> Tuple[float, float]:
    # dummy call - Validation would silence the exceptions
    learner(data)

    res: Results = TestOnTrainingData()(data, [learner])
    res_cv: Results = CrossValidation(k=N_FOLD)(data, [learner])
    # pylint: disable=unsubscriptable-object
    return R2(res)[0], R2(res_cv)[0]


def permutation(
        data: Table,
        learner: Learner,
        n_perm: int = 100,
        progress_callback: Callable = dummy_callback
) -> PermutationResults:
    r2, q2 = _validate(data, learner)
    r2_scores = [r2]
    q2_scores = [q2]
    correlations = [100.0]
    progress_callback(0, "Calculating...")
    np.random.seed(0)

    data_perm = data.copy()
    for i in range(n_perm):
        progress_callback(i / n_perm)
        np.random.shuffle(data_perm.Y)
        r2, q2 = _validate(data_perm, learner)
        correlations.append(_correlation(data.Y, data_perm.Y))
        r2_scores.append(r2)
        q2_scores.append(q2)

    correlations = np.abs(correlations)
    r2_res = linregress([correlations[0], np.mean(correlations[1:])],
                        [r2_scores[0], np.mean(r2_scores[1:])])
    q2_res = linregress([correlations[0], np.mean(correlations[1:])],
                        [q2_scores[0], np.mean(q2_scores[1:])])

    return (correlations,
            r2_scores, r2_res.intercept, r2_res.slope,
            q2_scores, q2_res.intercept, q2_res.slope,
            data.domain.class_var.name)


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
        self.setLabel(axis="left", text="R2, Q2")

    def _create_legend(self) -> LegendItem:
        legend = LegendItem()
        legend.setParentItem(self.getViewBox())
        legend.anchor((1, 1), (1, 1), offset=(-5, -5))
        legend.hide()
        return legend

    def set_data(
            self,
            corr: np.ndarray,
            r2_scores: List,
            r2_intercept: float,
            r2_slope: float,
            q2_scores: List,
            q2_intercept: float,
            q2_slope: float,
            name: str
    ):
        self.clear()
        title = f"{name} Intercepts: " \
                f"R2=(0.0, {round(r2_intercept, 4)}), " \
                f"Q2=(0.0, {round(q2_intercept, 4)})"
        self.setTitle(title)

        x = np.array([0, 100])
        pen = pg.mkPen("#000", width=2, style=Qt.DashLine)
        r2_line = pg.PlotCurveItem(x, r2_intercept + r2_slope * x, pen=pen)
        q2_line = pg.PlotCurveItem(x, q2_intercept + q2_slope * x, pen=pen)

        point_pen = pg.mkPen("#333")
        r2_kwargs = {"pen": point_pen, "symbol": "o", "brush": "#6fa255"}
        q2_kwargs = {"pen": point_pen, "symbol": "s", "brush": "#3a78b6"}

        kwargs = {"size": 12}
        kwargs.update(r2_kwargs)
        r2_points = pg.ScatterPlotItem(corr, r2_scores, **kwargs)
        kwargs.update(q2_kwargs)
        q2_points = pg.ScatterPlotItem(corr, q2_scores, **kwargs)

        self.addItem(r2_line)
        self.addItem(q2_line)
        self.addItem(r2_points)
        self.addItem(q2_points)

        self.legend.clear()
        self.legend.addItem(pg.ScatterPlotItem(**r2_kwargs), "R2")
        self.legend.addItem(pg.ScatterPlotItem(**q2_kwargs), "Q2")
        self.legend.show()


class OWPermutationPlot(OWWidget, ConcurrentWidgetMixin):
    name = "Permutation Plot"
    description = "Permutation analysis plotting R2 and Q2"
    icon = "icons/PermutationPlot.svg"
    priority = 1100

    n_permutations = Setting(100)
    visual_settings = Setting({}, schema_only=True)
    graph_name = "graph.plotItem"

    class Inputs:
        data = Input("Data", Table)
        learner = Input("Lerner", Learner)

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

    def _run(self):
        if self._data is None or self._learner is None:
            return
        self.start(run, self._data, self._learner, self.n_permutations)

    def on_done(self, result: PermutationResults):
        self.graph.set_data(*result)

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
        self.report_plot()

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        # pylint: disable=unsupported-assignment-operation
        self.visual_settings[key] = value


if __name__ == "__main__":
    from Orange.regression import LinearRegressionLearner

    housing = Table("housing")
    pls = LinearRegressionLearner()
    # permutation(housing, pls)

    WidgetPreview(OWPermutationPlot).run(
        set_data=housing, set_learner=pls)
