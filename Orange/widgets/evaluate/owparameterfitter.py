from typing import Optional, Callable, Collection, Sequence

import numpy as np
from AnyQt.QtCore import QPointF, Qt, QSize
from AnyQt.QtGui import QStandardItemModel, QStandardItem, \
    QPainter, QFontMetrics
from AnyQt.QtWidgets import QGraphicsSceneHelpEvent, QToolTip, \
    QGridLayout, QSizePolicy, QWidget

import pyqtgraph as pg

from orangewidget.utils.itemmodels import signal_blocking
from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog, \
    KeyType, ValueType

from Orange.base import Learner
from Orange.data import Table
from Orange.evaluation import CrossValidation, TestOnTrainingData, Results
from Orange.evaluation.scoring import Score, AUC, R2
from Orange.modelling import Fitter
from Orange.util import dummy_callback, wrap_callback
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import userinput
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
ScoreType = tuple[int, tuple[float, float]]
# scores, score name, label
FitterResults = tuple[list[ScoreType], str, str]


def _validate(
        data: Table,
        learner: Learner,
        scorer: type[Score],
        progress_callback: Callable
) -> tuple[float, float]:
    res: Results = TestOnTrainingData()(data, [learner],
                                        suppresses_exceptions=False,
                                        callback=wrap_callback(
                                            progress_callback, 0, 1/(1+N_FOLD))
                                        )
    res_cv: Results = CrossValidation(k=N_FOLD)(data, [learner],
                                                suppresses_exceptions=False,
                                                callback=wrap_callback(
                                                    progress_callback, 1/(1+N_FOLD), 1.)
                                                )
    # pylint: disable=unsubscriptable-object
    return scorer(res)[0], scorer(res_cv)[0]


def _search(
        data: Table,
        learner: Learner,
        fitted_parameter_props: Learner.FittedParameter,
        initial_parameters: dict[str, int],
        steps: Collection[int],
        progress_callback: Callable = dummy_callback
) -> FitterResults:
    progress_callback(0, "Calculating...")
    scores = []
    scorer = AUC if data.domain.has_discrete_class else R2
    name = fitted_parameter_props.name
    for i, value in enumerate(steps):
        params = initial_parameters.copy()
        params[name] = value
        result = _validate(data, type(learner)(**params), scorer,
                           wrap_callback(progress_callback, i / len(steps), (i+1) / len(steps)))
        scores.append((value, result))
    return scores, scorer.name, fitted_parameter_props.label


def run(
        data: Table,
        learner: Learner,
        fitted_parameter_props: Learner.FittedParameter,
        initial_parameters: dict[str, int],
        steps: Collection[int],
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
        self.grid_settings: Optional[dict] = None
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
                x=False, y=self.grid_settings[self.SHOW_GRID_LABEL],
                alpha=self.grid_settings[Updater.ALPHA_LABEL] / 255)

        self._setters[self.PLOT_BOX] = {self.GRID_LABEL: update_grid}

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
        self.__bar_item_tr: Optional[pg.BarGraphItem] = None
        self.__bar_item_cv: Optional[pg.BarGraphItem] = None
        self.__data: Optional[list[ScoreType]] = None
        self.legend = self._create_legend()
        self.parameter_setter = ParameterSetter(self)
        self.setMouseEnabled(False, False)
        self.hideButtons()

        self.showGrid(x=False, y=self.parameter_setter.DEFAULT_SHOW_GRID,
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
        self.setLabel(axis="bottom", text=None)
        self.setLabel(axis="left", text=None)
        self.getAxis("bottom").setTicks(None)

    def set_data(
            self,
            scores: list[ScoreType],
            score_name: str,
            parameter_name: str
    ):
        self.__data = scores
        self.clear()
        self.setLabel(axis="bottom", text=parameter_name)
        self.setLabel(axis="left", text=score_name)

        ticks = [[(i, str(val)) for i, (val, _)
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
            text = "<table align=left>" \
                   "<tr>" \
                   "<td><b>Train:</b></td>" \
                   f"<td>{round(scores[0], 3)}</td>" \
                   "</tr><tr>" \
                   "<td><b>CV:</b></td>" \
                   f"<td>{round(scores[1], 3)}</td>" \
                   "</tr>" \
                   "</table>"
        if text:
            QToolTip.showText(ev.screenPos(), text, widget=self)
            return True
        else:
            return False

    def __get_index_at(self, point: QPointF) -> Optional[int]:
        x = point.x()
        index = round(x)
        # pylint: disable=unsubscriptable-object
        heights_tr: list = self.__bar_item_tr.opts["height"]
        heights_cv: list = self.__bar_item_cv.opts["height"]
        if 0 <= index < len(heights_tr) and abs(index - x) <= self.BAR_WIDTH:
            if index > x and 0 <= point.y() <= heights_tr[index]:
                return index
            if x > index and 0 <= point.y() <= heights_cv[index]:
                return index
        return None


class RangePreview(QWidget):
    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(font.pointSize() - 3)
        self.setFont(font)

        self.__steps: Optional[Sequence[int]] = None
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

    def minimumSizeHint(self):
        return QSize(1, 20)

    def set_steps(self, steps: Optional[Sequence[int]]):
        self.__steps = steps
        self.update()

    def steps(self):
        return self.__steps

    def paintEvent(self, _):
        if not self.__steps:
            return
        painter = QPainter(self)
        metrics = QFontMetrics(self.font())
        style = self.style()
        rect = self.rect()

        # Indent by the width of the radio button indicator
        rect.adjust(style.pixelMetric(style.PM_IndicatorWidth)
                    + style.pixelMetric(style.PM_CheckBoxLabelSpacing), 0, 0, 0)

        last_text = f"{self.__steps[-1]}"
        if len(self.__steps) > 1:
            last_text = ", " + last_text
        last_width = metrics.horizontalAdvance(last_text)

        elided_text = metrics.elidedText(
            "Steps: " + ", ".join(map(str, self.__steps[:-1])),
            Qt.ElideRight, rect.width() - last_width)
        elided_width = metrics.horizontalAdvance(elided_text)

        # Right-align by indenting by the underflow width
        rect.adjust(rect.width() - elided_width - last_width, 0, 0, 0)

        painter.drawText(rect, Qt.AlignLeft, elided_text)
        rect.adjust(elided_width, 0, 0, 0)
        painter.drawText(rect, Qt.AlignLeft, last_text)


class OWParameterFitter(OWWidget, ConcurrentWidgetMixin):
    name = "Parameter Fitter"
    description = "Fit learner for various values of fitting parameter."
    icon = "icons/ParameterFitter.svg"
    priority = 1110
    keywords = "parameter, fitter, tuning"

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
    type: int = Setting(FROM_RANGE)
    minimum: int = Setting(DEFAULT_MINIMUM, schema_only=True)
    maximum: int = Setting(DEFAULT_MAXIMUM, schema_only=True)
    manual_steps: str = Setting("", schema_only=True)
    auto_commit = Setting(True)

    class Error(OWWidget.Error):
        unknown_err = Msg("{}")
        not_enough_data = Msg(f"At least {N_FOLD} instances are needed.")
        incompatible_learner = Msg("{}")
        manual_steps_error = Msg("Invalid values for '{}': {}")
        min_max_error = Msg("Minimum must be less than maximum.")
        missing_target = Msg("Data has no target.")

    class Warning(OWWidget.Warning):
        no_parameters = Msg("{} has no parameters to fit.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self._data: Optional[Table] = None
        self._learner: Optional[Learner] = None
        self.__parameters_model = QStandardItemModel()
        self.__initialize_settings = False

        self.setup_gui()
        VisualSettingsDialog(
            self, self.graph.parameter_setter.initial_settings
        )

    def setup_gui(self):
        self._add_plot()
        self._add_controls()

    def _add_plot(self):
        # This is a part of __init__
        # pylint: disable=attribute-defined-outside-init
        box = gui.vBox(self.mainArea)
        self.graph = FitterPlot()
        box.layout().addWidget(self.graph)

    def _add_controls(self):
        # This is a part of __init__
        # pylint: disable=attribute-defined-outside-init
        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Settings", orientation=layout)
        self.__combo = gui.comboBox(None, self, "parameter_index",
                                    model=self.__parameters_model,
                                    callback=self.__on_parameter_changed)
        layout.addWidget(self.__combo, 0, 0, 1, 2)

        buttons = gui.radioButtons(None, self, "type",
                                   callback=self.__on_type_changed)
        button = gui.appendRadioButton(buttons, "Range:")
        layout.addWidget(button, 1, 0)

        # pylint: disable=use-dict-literal
        kw = dict(minv=-MIN_MAX_SPIN, maxv=MIN_MAX_SPIN,
                  alignment=Qt.AlignRight,
                  callback=self.__on_min_max_changed)
        box = gui.hBox(None)
        self.__spin_min = gui.spin(box, self, "minimum", label="From:", **kw)
        layout.addWidget(box, 1, 1)

        box = gui.hBox(None)
        self.__spin_max = gui.spin(box, self, "maximum", label="To:", **kw)
        layout.addWidget(box, 2, 1)

        self.range_preview = RangePreview()
        layout.addWidget(self.range_preview, 3, 0, 1, 2)

        gui.appendRadioButton(buttons, "Manual:")
        layout.addWidget(buttons, 4, 0)
        self.edit = gui.lineEdit(None, self, "manual_steps",
                                 placeholderText="e.g. 10, 20, ..., 50",
                                 callback=self.__on_manual_changed)
        layout.addWidget(self.edit, 4, 1)

        # gui.lineEdit's connect does not call the callback on return pressed
        # if the line hasn't changed.
        @self.edit.returnPressed.connect
        def _():
            if self.type != self.MANUAL:
                self.type = self.MANUAL
                self.__on_type_changed()

        gui.rubber(self.controlArea)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

        self._update_preview()

    def __on_type_changed(self):
        self._settings_changed()

    def __on_parameter_changed(self):
        self.__initialize_settings = True
        self._set_range_controls(self.fitted_parameters[self.parameter_index])
        self._settings_changed()

    def __on_min_max_changed(self):
        self.type = self.FROM_RANGE
        self._settings_changed()

    def __on_manual_changed(self):
        self.type = self.MANUAL
        self._settings_changed()

    def _settings_changed(self):
        self._update_preview()
        self.commit.deferred()

    @property
    def fitted_parameters(self) -> list:
        if not self._learner:
            return []
        return self._learner.fitted_parameters

    @property
    def initial_parameters(self) -> dict:
        if not self._learner:
            return {}
        if isinstance(self._learner, Fitter):
            return self._learner.get_params(self._data or "classification")
        return self._learner.params

    @property
    def steps(self) -> tuple[int, ...]:
        self.Error.min_max_error.clear()
        self.Error.manual_steps_error.clear()

        if self.type == self.FROM_RANGE:
            return self._steps_from_range()
        else:
            return self._steps_from_manual()

    def _steps_from_range(self) -> tuple[int, ...]:
        if self.maximum < self.minimum:
            self.Error.min_max_error()
            return ()

        if self.minimum == self.maximum:
            return (self.minimum, )

        diff = self.maximum - self.minimum
        # This should give between 10 and 15 steps
        exp = max(0, int(np.ceil(np.log10(diff / 1.5))) - 1)
        step = int(10 ** exp)
        return (self.minimum,
                *range((self.minimum // step + 1) * step, self.maximum, step),
                self.maximum)

    def _steps_from_manual(self) -> tuple[int, ...]:
        param = self.fitted_parameters[self.parameter_index]
        try:
            steps = userinput.numbers_from_list(
                self.manual_steps, int, param.min, param.max)
        except ValueError as ex:
            self.Error.manual_steps_error(param.label, ex)
            return ()
        if steps and "..." not in self.manual_steps:
            self.manual_steps = ", ".join(map(str, steps))
        return steps

    @Inputs.data
    @check_multiple_targets_input
    def set_data(self, data: Optional[Table]):
        self.Error.not_enough_data.clear()
        self.Error.missing_target.clear()
        self._data = data
        if self._data and len(self._data) < N_FOLD:
            self.Error.not_enough_data()
            self._data = None
        if self._data and len(self._data.domain.class_vars) < 1:
            self.Error.missing_target()
            self._data = None

    @Inputs.learner
    def set_learner(self, learner: Optional[Learner]):
        self.Warning.clear()
        self.Error.manual_steps_error.clear()
        self.Error.min_max_error.clear()
        self.__parameters_model.clear()

        if not learner:
            self.__initialize_settings = False
            # reset spin controls
            ars = (None, None, int, None, None)
            self._set_range_controls(Learner.FittedParameter(*ars))

        elif self._learner:
            self.__initialize_settings = \
                learner.fitted_parameters != self.fitted_parameters

        else:
            # changed by user or opened workflow
            self.__initialize_settings = \
                self.parameter_index == self.DEFAULT_PARAMETER_INDEX and \
                self.minimum == self.DEFAULT_MINIMUM and \
                self.maximum == self.DEFAULT_MAXIMUM

        self._learner = learner
        if self._learner is None:
            return

        for param in self.fitted_parameters:
            item = QStandardItem(param.label)
            self.__parameters_model.appendRow(item)

        if not self.fitted_parameters:
            self.Warning.no_parameters(self._learner.name)
        else:
            if self.__initialize_settings:
                self.parameter_index = 0
            else:
                self.__combo.setCurrentIndex(self.parameter_index)
            self._set_range_controls(
                self.fitted_parameters[self.parameter_index])

        self._update_preview()

    def handleNewSignals(self):
        self.Error.unknown_err.clear()
        self.Error.incompatible_learner.clear()
        self.clear()

        if not self._data or not self._learner:
            return

        reason = self._learner.incompatibility_reason(self._data.domain)
        if reason:
            self.Error.incompatible_learner(reason)
            return

        self.commit.now()

    def _set_range_controls(self, param: Learner.FittedParameter):
        assert param.type == int, \
            "The widget currently supports only int parameters"

        # Block signals to avoid changing `self.type`
        with signal_blocking(self.__spin_min), signal_blocking(self.__spin_max):
            if param.min is not None:
                self.__spin_min.setMinimum(param.min)
                self.__spin_max.setMinimum(param.min)
                self.minimum = param.min if self.__initialize_settings else \
                    max(self.minimum, param.min)
            else:
                self.__spin_min.setMinimum(-MIN_MAX_SPIN)
                self.__spin_max.setMinimum(-MIN_MAX_SPIN)
                if self.__initialize_settings:
                    self.minimum = self.initial_parameters[param.name]
            if param.max is not None:
                self.__spin_min.setMaximum(param.max)
                self.__spin_max.setMaximum(param.max)
                if self.__initialize_settings:
                    self.maximum = param.max
                self.maximum = param.max if self.__initialize_settings else \
                    min(self.maximum, param.max)
            else:
                self.__spin_min.setMaximum(MIN_MAX_SPIN)
                self.__spin_max.setMaximum(MIN_MAX_SPIN)
                if self.__initialize_settings:
                    self.maximum = self.initial_parameters[param.name]
            self.__initialize_settings = False

        tip = "Enter a list of values"
        if param.min is not None:
            if param.max is not None:
                self.edit.setToolTip(f"{tip} between {param.min} and {param.max}.")
            else:
                self.edit.setToolTip(f"{tip} greater or equal to {param.min}.")
        elif param.max is not None:
            self.edit.setToolTip(f"{tip} smaller or equal to {param.max}.")
        else:
            self.edit.setToolTip("")

    def _update_preview(self):
        if self.type == self.FROM_RANGE:
            self.range_preview.set_steps(self.steps)
        else:
            self.range_preview.set_steps(None)

    def clear(self):
        self.cancel()
        self.graph.clear_all()

    @gui.deferred
    def commit(self):
        self.graph.clear_all()
        if self._data is None or self._learner is None or \
                not self.fitted_parameters or not self.steps:
            return
        self.start(run, self._data, self._learner,
                   self.fitted_parameters[self.parameter_index],
                   self.initial_parameters, self.steps)

    def on_done(self, result: FitterResults):
        self.graph.set_data(*result)

    def on_exception(self, ex: Exception):
        self.Error.unknown_err(ex)

    def on_partial_result(self, _):
        pass

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def send_report(self):
        if self._data is None or self._learner is None \
                or not self.fitted_parameters:
            return
        parameter = self.fitted_parameters[self.parameter_index].label
        self.report_items("Settings",
                          [("Parameter", parameter),
                           ("Range", ", ".join(map(str, self.steps)))])
        self.report_name("Plot")
        self.report_plot()

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        # pylint: disable=unsupported-assignment-operation
        self.visual_settings[key] = value


if __name__ == "__main__":
    from Orange.regression import PLSRegressionLearner

    WidgetPreview(OWParameterFitter).run(
        set_data=Table("housing"), set_learner=PLSRegressionLearner())
