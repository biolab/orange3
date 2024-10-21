from typing import Optional, Tuple, Callable, List, Dict, Iterable, Sized

import numpy as np
from AnyQt.QtCore import QPointF, Qt
from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtWidgets import QGraphicsSceneHelpEvent, QToolTip, QSpinBox, \
    QComboBox

import pyqtgraph as pg

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog, \
    KeyType, ValueType

from Orange.base import Learner
from Orange.data import Table
from Orange.data.table import DomainTransformationError
from Orange.evaluation import CrossValidation, TestOnTrainingData, Results
from Orange.evaluation.scoring import Score, AUC, R2
from Orange.modelling import Fitter
from Orange.util import dummy_callback
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.multi_target import check_multiple_targets_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import LegendItem
from Orange.widgets.visualize.utils.customizableplot import \
    CommonParameterSetter, Updater
from Orange.widgets.visualize.utils.plotutils import PlotWidget, \
    HelpEventDelegate
from Orange.widgets.widget import OWWidget, Input, Msg

N_FOLD = 7
MIN_MAX_SPIN = 100000
ScoreType = Tuple[int, Tuple[float, float]]
# scores, score name, tick label
FitterResults = Tuple[List[ScoreType], str, str]


def _validate(
        data: Table,
        learner: Learner,
        scorer: type[Score]
) -> Tuple[float, float]:
    # dummy call - Validation would silence the exceptions
    learner(data)

    res: Results = TestOnTrainingData()(data, [learner])
    res_cv: Results = CrossValidation(k=N_FOLD)(data, [learner])
    # pylint: disable=unsubscriptable-object
    return scorer(res)[0], scorer(res_cv)[0]


def _search(
        data: Table,
        learner: Learner,
        fitted_parameter_props: Learner.FittedParameter,
        initial_parameters: Dict,
        steps: Sized,
        progress_callback: Callable = dummy_callback
) -> FitterResults:
    progress_callback(0, "Calculating...")
    scores = []
    scorer = AUC if data.domain.has_discrete_class else R2
    parameter_name = fitted_parameter_props.parameter_name
    for i, value in enumerate(steps):
        progress_callback(i / len(steps))
        params = initial_parameters.copy()
        params[parameter_name] = value
        result = _validate(data, type(learner)(**params), scorer)
        scores.append((value, result))
    return scores, scorer.name, fitted_parameter_props.tick_label


def run(
        data: Table,
        learner: Learner,
        fitted_parameter_props: Learner.FittedParameter,
        initial_parameters: Dict,
        steps: Sized,
        state: TaskState
) -> FitterResults:
    def callback(i: float, status: str = ""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            # pylint: disable=broad-exception-raised
            raise Exception

    return _search(data, learner, fitted_parameter_props, initial_parameters,
                   steps, callback)


class ParameterSetter(CommonParameterSetter):
    GRID_LABEL, SHOW_GRID_LABEL = "Gridlines", "Show"
    DEFAULT_ALPHA_GRID, DEFAULT_SHOW_GRID = 80, True

    def __init__(self, master):
        self.grid_settings: Dict = None
        self.master: FitterPlot = master
        super().__init__()

    def update_setters(self):
        self.grid_settings = {
            Updater.ALPHA_LABEL: self.DEFAULT_ALPHA_GRID,
            self.SHOW_GRID_LABEL: self.DEFAULT_SHOW_GRID,
        }

        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
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


class FitterPlot(PlotWidget):
    BAR_WIDTH = 0.4

    def __init__(self):
        super().__init__(enableMenu=False)
        self.__bar_item_tr: pg.BarGraphItem = None
        self.__bar_item_cv: pg.BarGraphItem = None
        self.__data: List[ScoreType] = None
        self.legend = self._create_legend()
        self.parameter_setter = ParameterSetter(self)
        self.setMouseEnabled(False, False)
        self.hideButtons()

        self.showGrid(False, True)
        self.showGrid(y=self.parameter_setter.DEFAULT_SHOW_GRID,
                      alpha=self.parameter_setter.DEFAULT_ALPHA_GRID / 255)

        self.tooltip_delegate = HelpEventDelegate(self.help_event)
        self.scene().installEventFilter(self.tooltip_delegate)

    def _create_legend(self) -> LegendItem:
        legend = LegendItem()
        legend.setParentItem(self.getViewBox())
        legend.anchor((1, 1), (1, 1), offset=(-5, -5))
        legend.hide()
        return legend

    def clear_all(self):
        self.clear()
        self.__bar_item_tr = None
        self.__bar_item_cv = None
        self.__data = None
        self.setLabel(axis="left", text=None)
        self.getAxis("bottom").setTicks(None)

    def set_data(
            self,
            scores: List[ScoreType],
            score_name: str,
            tick_name: str
    ):
        self.__data = scores
        self.clear()
        self.setLabel(axis="bottom", text=" ")
        self.setLabel(axis="left", text=score_name)

        ticks = [[(i, f"{tick_name}[{val}]") for i, (val, _)
                  in enumerate(scores)]]
        self.getAxis("bottom").setTicks(ticks)

        brush_tr = "#6fa255"
        brush_cv = "#3a78b6"
        pen = pg.mkPen("#333")
        kwargs = {"pen": pen, "width": self.BAR_WIDTH}
        bar_item_tr = pg.BarGraphItem(x=np.arange(len(scores)) - 0.2,
                                      height=[(s[0]) for _, s in scores],
                                      brush=brush_tr, **kwargs)
        bar_item_cv = pg.BarGraphItem(x=np.arange(len(scores)) + 0.2,
                                      height=[(s[1]) for _, s in scores],
                                      brush=brush_cv, **kwargs)
        self.addItem(bar_item_tr)
        self.addItem(bar_item_cv)
        self.__bar_item_tr = bar_item_tr
        self.__bar_item_cv = bar_item_cv

        self.legend.clear()
        kwargs = {"pen": pen, "symbol": "s"}
        scatter_item_tr = pg.ScatterPlotItem(brush=brush_tr, **kwargs)
        scatter_item_cv = pg.ScatterPlotItem(brush=brush_cv, **kwargs)
        self.legend.addItem(scatter_item_tr, "Train")
        self.legend.addItem(scatter_item_cv, "CV")
        Updater.update_legend_font(self.legend.items,
                                   **self.parameter_setter.legend_settings)
        self.legend.show()

    def help_event(self, ev: QGraphicsSceneHelpEvent) -> bool:
        if self.__bar_item_tr is None:
            return False

        pos = self.__bar_item_tr.mapFromScene(ev.scenePos())
        index = self.__get_index_at(pos)
        text = ""
        if index is not None:
            _, scores = self.__data[index]
            text = f"<table align=left>" \
                   f"<tr>" \
                   f"<td><b>Train:</b></td>" \
                   f"<td>{round(scores[0], 3)}</td>" \
                   f"</tr><tr>" \
                   f"<td><b>CV:</b></td>" \
                   f"<td>{round(scores[1], 3)}</td>" \
                   f"</tr>" \
                   f"</table>"
        if text:
            QToolTip.showText(ev.screenPos(), text, widget=self)
            return True
        else:
            return False

    def __get_index_at(self, point: QPointF) -> Optional[int]:
        x = point.x()
        index = round(x)
        # pylint: disable=unsubscriptable-object
        heights_tr: List = self.__bar_item_tr.opts["height"]
        heights_cv: List = self.__bar_item_cv.opts["height"]
        if 0 <= index < len(heights_tr) and abs(index - x) <= self.BAR_WIDTH:
            if index > x and 0 <= point.y() <= heights_tr[index]:
                return index
            if x > index and 0 <= point.y() <= heights_cv[index]:
                return index
        return None


class OWParameterFitter(OWWidget, ConcurrentWidgetMixin):
    name = "Parameter Fitter"
    description = "Fit learner for various values of fitting parameter."
    icon = "icons/ParameterFitter.svg"
    priority = 1110

    visual_settings = Setting({}, schema_only=True)
    graph_name = "graph.plotItem"

    class Inputs:
        data = Input("Data", Table)
        learner = Input("Learner", Learner)

    DEFAULT_PARAMETER_INDEX = 0
    DEFAULT_MINIMUM = 1
    DEFAULT_MAXIMUM = 9
    parameter_index = Setting(DEFAULT_PARAMETER_INDEX, schema_only=True)
    FROM_RANGE, MANUAL = range(2)
    type = Setting(FROM_RANGE)
    minimum = Setting(DEFAULT_MINIMUM, schema_only=True)
    maximum = Setting(DEFAULT_MAXIMUM, schema_only=True)
    manual_steps = Setting("", schema_only=True)
    auto_commit = Setting(True)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")
        not_enough_data = Msg(f"At least {N_FOLD} instances are needed.")
        incompatible_learner = Msg("{}")

    class Warning(OWWidget.Warning):
        no_parameters = Msg("{} has no parameters to fit.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self._data: Optional[Table] = None
        self._learner: Optional[Learner] = None
        self.graph: FitterPlot = None
        self.__parameters_model = QStandardItemModel()
        self.__combo: QComboBox = None
        self.__spin_min: QSpinBox = None
        self.__spin_max: QSpinBox = None
        self.preview: str = ""

        self.__pending_parameter_index = self.parameter_index \
            if self.parameter_index != self.DEFAULT_PARAMETER_INDEX else None
        self.__pending_minimum = self.minimum \
            if self.minimum != self.DEFAULT_MINIMUM else None
        self.__pending_maximum = self.maximum \
            if self.maximum != self.DEFAULT_MAXIMUM else None

        self.setup_gui()
        VisualSettingsDialog(
            self, self.graph.parameter_setter.initial_settings
        )

    def setup_gui(self):
        self._add_plot()
        self._add_controls()

    def _add_plot(self):
        box = gui.vBox(self.mainArea)
        self.graph = FitterPlot()
        box.layout().addWidget(self.graph)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Settings")
        self.__combo = gui.comboBox(box, self, "parameter_index",
                                    model=self.__parameters_model,
                                    callback=self.__on_parameter_changed)

        buttons = gui.radioButtons(box, self, "type",
                                   callback=self.__on_setting_changed)

        gui.appendRadioButton(buttons, "Range")
        hbox = gui.indentedBox(buttons, 20, orientation=Qt.Horizontal)
        kw = {"minv": -MIN_MAX_SPIN, "maxv": MIN_MAX_SPIN,
              "callback": self.__on_setting_changed}
        self.__spin_min = gui.spin(hbox, self, "minimum", label="Min:", **kw)
        self.__spin_max = gui.spin(hbox, self, "maximum", label="Max:", **kw)

        gui.appendRadioButton(buttons, "Manual")
        hbox = gui.indentedBox(box, 20, orientation=Qt.Horizontal)
        gui.lineEdit(hbox, self, "manual_steps", placeholderText="10, 20, 30",
                     callback=self.__on_setting_changed)

        box = gui.vBox(self.controlArea, "Steps preview")
        self.preview = ""
        gui.label(box, self, "%(preview)s", wordWrap=True)

        gui.rubber(self.controlArea)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    def __on_parameter_changed(self):
        self._set_range_controls()
        self.__on_setting_changed()

    def __on_setting_changed(self):
        self._update_preview()
        self.commit.deferred()

    @property
    def fitted_parameters(self) -> List:
        if not self._learner or not self._data:
            return []
        return self._learner.fitted_parameters(self._data) \
            if isinstance(self._learner, Fitter) \
            else self._learner.fitted_parameters()

    @property
    def initial_parameters(self) -> Dict:
        if not self._learner or not self._data:
            return {}
        return self._learner.get_params(self._data) \
            if isinstance(self._learner, Fitter) else self._learner.params

    @property
    def steps(self) -> Iterable[int]:
        if self.type == self.FROM_RANGE:
            step = 1
            diff = self.maximum - self.minimum
            if diff > 0:
                exp = int(np.ceil(np.log10(diff + 1))) - 1
                step = int(10 ** exp)
            return range(self.minimum, self.maximum + step, step)
        else:
            try:
                return [int(s) for s in self.manual_steps.split(",")]
            except ValueError:
                return []

    @Inputs.data
    @check_multiple_targets_input
    def set_data(self, data: Optional[Table]):
        self.Error.not_enough_data.clear()
        self._data = data
        if self._data and len(self._data) < N_FOLD:
            self.Error.not_enough_data()
            self._data = None

    @Inputs.learner
    def set_learner(self, learner: Optional[Learner]):
        self._learner = learner

    def handleNewSignals(self):
        self.Warning.no_parameters.clear()
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

        for param in self.fitted_parameters:
            item = QStandardItem(param.label)
            self.__parameters_model.appendRow(item)
        if not self.fitted_parameters:
            self.Warning.no_parameters(self._learner.name)
        else:
            if self.__pending_parameter_index is not None:
                self.parameter_index = self.__pending_parameter_index
                self.__combo.setCurrentIndex(self.parameter_index)
                self.__pending_parameter_index = None
            self._set_range_controls()
            if self.__pending_minimum is not None:
                self.minimum = self.__pending_minimum
                self.__pending_minimum = None
            if self.__pending_maximum is not None:
                self.maximum = self.__pending_maximum
                self.__pending_maximum = None

        self._update_preview()
        self.commit.now()

    def _set_range_controls(self):
        param = self.fitted_parameters[self.parameter_index]

        assert param.type == int

        if param.min is not None:
            self.__spin_min.setMinimum(param.min)
            self.__spin_max.setMinimum(param.min)
            self.minimum = param.min
        else:
            self.__spin_min.setMinimum(-MIN_MAX_SPIN)
            self.__spin_max.setMinimum(-MIN_MAX_SPIN)
            self.minimum = self.initial_parameters[param.parameter_name]
        if param.max is not None:
            self.__spin_min.setMaximum(param.min)
            self.__spin_max.setMaximum(param.min)
            self.maximum = param.max
        else:
            self.__spin_min.setMaximum(MIN_MAX_SPIN)
            self.__spin_max.setMaximum(MIN_MAX_SPIN)
            self.maximum = self.initial_parameters[param.parameter_name]

    def _update_preview(self):
        self.preview = str(list(self.steps))

    def clear(self):
        self.cancel()
        self.graph.clear_all()
        self.__parameters_model.clear()

    @gui.deferred
    def commit(self):
        if self._data is None or self._learner is None or not \
                self.fitted_parameters:
            return
        self.graph.clear_all()
        self.start(run, self._data, self._learner,
                   self.fitted_parameters[self.parameter_index],
                   self.initial_parameters, self.steps)

    def on_done(self, result: FitterResults):
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

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        # pylint: disable=unsupported-assignment-operation
        self.visual_settings[key] = value


if __name__ == "__main__":
    from Orange.regression import PLSRegressionLearner

    WidgetPreview(OWParameterFitter).run(
        set_data=Table("housing"), set_learner=PLSRegressionLearner())
