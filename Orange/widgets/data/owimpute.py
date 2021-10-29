import copy
import logging
import enum
import concurrent.futures
from concurrent.futures import Future
from collections import namedtuple, OrderedDict
from typing import List, Any, Dict, Tuple, Type, Optional

import numpy as np

from AnyQt.QtWidgets import (
    QGroupBox, QRadioButton, QPushButton, QHBoxLayout, QGridLayout,
    QVBoxLayout, QStackedWidget, QComboBox, QWidget,
    QButtonGroup, QStyledItemDelegate, QListView, QLabel
)
from AnyQt.QtCore import Qt, QThread, QModelIndex, QDateTime
from AnyQt.QtCore import pyqtSlot as Slot

from orangewidget.utils.listview import ListViewSearch

import Orange.data
from Orange.preprocess import impute
from Orange.base import Learner
from Orange.widgets import gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils import concurrent as qconcurrent
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.spinbox import DoubleSpinBox
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.classification import SimpleTreeLearner

DisplayMethodRole = Qt.UserRole
StateRole = DisplayMethodRole + 0xf4


class DisplayFormatDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        method = index.data(DisplayMethodRole)
        var = index.model()[index.row()]
        if method:
            option.text = method.format_variable(var)

            if not method.supports_variable(var):
                option.palette.setColor(option.palette.Text, Qt.darkRed)

            if isinstance(getattr(method, 'method', method), impute.DoNotImpute):
                option.palette.setColor(option.palette.Text, Qt.darkGray)


class AsDefault(impute.BaseImputeMethod):
    name = "Default (above)"
    short_name = ""
    format = "{var.name}"
    columns_only = True

    method = impute.DoNotImpute()

    def __getattr__(self, item):
        return getattr(self.method, item)

    def supports_variable(self, variable):
        return self.method.supports_variable(variable)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


class SparseNotSupported(ValueError):
    pass


class VariableNotSupported(ValueError):
    pass


RowMask = namedtuple("RowMask", ["mask"])


class Task:
    futures = []    # type: List[Future]
    watcher = ...   # type: qconcurrent.FutureSetWatcher
    cancelled = False

    def __init__(self, futures, watcher):
        self.futures = futures
        self.watcher = watcher

    def cancel(self):
        self.cancelled = True
        for f in self.futures:
            f.cancel()


class Method(enum.IntEnum):
    AsAboveSoBelow = 0  # a special
    Leave, Average, AsValue, Model, Random, Drop, Default = range(1, 8)


METHODS = OrderedDict([
    (Method.AsAboveSoBelow, AsDefault),
    (Method.Leave, impute.DoNotImpute),
    (Method.Average, impute.Average),
    (Method.AsValue, impute.AsValue),
    (Method.Model, impute.Model),
    (Method.Random, impute.Random),
    (Method.Drop, impute.DropInstances),
    (Method.Default, impute.Default),
])  # type: Dict[Method, Type[impute.BaseImputeMethod]]


#: variable state type: store effective imputation method and optional
#: parameters (currently used for storing per variable `impute.Default`'s
#: value). NOTE: Must be hashable and __eq__ comparable.
State = Tuple[int, Tuple[Any, ...]]
#: variable key, for storing/restoring state
VarKey = Tuple[str, str]

VariableState = Dict[VarKey, State]


def var_key(var):
    # type: (Orange.data.Variable) -> Tuple[str, str]
    qname = "{}.{}".format(type(var).__module__, type(var).__name__)
    return qname, var.name


DBL_MIN = np.finfo(float).min
DBL_MAX = np.finfo(float).max


class OWImpute(OWWidget):
    name = "Impute"
    description = "Impute missing values in the data table."
    icon = "icons/Impute.svg"
    priority = 2130
    keywords = ["substitute", "missing"]

    class Inputs:
        data = Input("Data", Orange.data.Table)
        learner = Input("Learner", Learner)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    class Error(OWWidget.Error):
        imputation_failed = Msg("Imputation failed for '{}'")
        model_based_imputer_sparse = \
            Msg("Model based imputer does not work for sparse data")

    class Warning(OWWidget.Warning):
        cant_handle_var = Msg("Default method can not handle '{}'")

    settingsHandler = settings.DomainContextHandler()

    _default_method_index = settings.Setting(int(Method.Leave))  # type: int
    # Per-variable imputation state (synced in storeSpecificSettings)
    _variable_imputation_state = settings.ContextSetting({})  # type: VariableState

    autocommit = settings.Setting(True)
    default_numeric_value = settings.Setting(0.0)
    default_time = settings.Setting(0)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Orange.data.Table]
        self.learner = None  # type: Optional[Learner]
        self.default_learner = SimpleTreeLearner(min_instances=10, max_depth=10)
        self.modified = False
        self.executor = qconcurrent.ThreadExecutor(self)
        self.__task = None

        main_layout = self.controlArea.layout()

        box = gui.vBox(self.controlArea, "Default Method")

        box_layout = QGridLayout()
        box_layout.setSpacing(8)
        box.layout().addLayout(box_layout)

        button_group = QButtonGroup()
        button_group.buttonClicked[int].connect(self.set_default_method)

        for i, (method, _) in enumerate(list(METHODS.items())[1:-1]):
            imputer = self.create_imputer(method)
            button = QRadioButton(imputer.name)
            button.setChecked(method == self.default_method_index)
            button_group.addButton(button, method)
            box_layout.addWidget(button, i % 3, i // 3)

        def set_default_time(datetime):
            datetime = datetime.toSecsSinceEpoch()
            if datetime != self.default_time:
                self.default_time = datetime
                if self.default_method_index == Method.Default:
                    self._invalidate()

        hlayout = QHBoxLayout()
        box.layout().addLayout(hlayout)
        button = QRadioButton("Fixed values; numeric variables:")
        button_group.addButton(button, Method.Default)
        button.setChecked(Method.Default == self.default_method_index)
        hlayout.addWidget(button)

        self.numeric_value_widget = DoubleSpinBox(
            minimum=DBL_MIN, maximum=DBL_MAX, singleStep=.1,
            value=self.default_numeric_value,
            alignment=Qt.AlignRight,
            enabled=self.default_method_index == Method.Default,
        )
        self.numeric_value_widget.editingFinished.connect(
            self.__on_default_numeric_value_edited
        )
        self.connect_control(
            "default_numeric_value", self.numeric_value_widget.setValue
        )
        hlayout.addWidget(self.numeric_value_widget)

        hlayout.addWidget(QLabel(", time:"))

        self.time_widget = gui.DateTimeEditWCalendarTime(self)
        self.time_widget.setEnabled(self.default_method_index == Method.Default)
        self.time_widget.setKeyboardTracking(False)
        self.time_widget.setContentsMargins(0, 0, 0, 0)
        self.time_widget.set_datetime(
            QDateTime.fromSecsSinceEpoch(self.default_time)
        )
        self.connect_control(
            "default_time",
            lambda value: self.time_widget.set_datetime(
                QDateTime.fromSecsSinceEpoch(value)
            )
        )
        self.time_widget.dateTimeChanged.connect(set_default_time)
        hlayout.addWidget(self.time_widget)

        self.default_button_group = button_group

        box = gui.hBox(self.controlArea, self.tr("Individual Attribute Settings"),
                       flat=False)

        self.varview = ListViewSearch(
            selectionMode=QListView.ExtendedSelection,
            uniformItemSizes=True
        )
        self.varview.setItemDelegate(DisplayFormatDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._on_var_selection_changed
        )
        self.selection = self.varview.selectionModel()

        box.layout().addWidget(self.varview)
        vertical_layout = QVBoxLayout(margin=0)

        self.methods_container = QWidget(enabled=False)
        method_layout = QVBoxLayout(margin=0)
        self.methods_container.setLayout(method_layout)

        button_group = QButtonGroup()
        for method in Method:
            imputer = self.create_imputer(method)
            button = QRadioButton(text=imputer.name)
            button_group.addButton(button, method)
            method_layout.addWidget(button)

        self.value_combo = QComboBox(
            minimumContentsLength=8,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            activated=self._on_value_selected
            )
        self.value_double = DoubleSpinBox(
            editingFinished=self._on_value_selected,
            minimum=DBL_MIN, maximum=DBL_MAX, singleStep=.1,
            )
        self.value_stack = value_stack = QStackedWidget()
        value_stack.addWidget(self.value_combo)
        value_stack.addWidget(self.value_double)
        method_layout.addWidget(value_stack)

        button_group.buttonClicked[int].connect(
            self.set_method_for_current_selection
        )

        self.reset_button = QPushButton(
            "Restore All to Default", enabled=False, default=False,
            autoDefault=False, clicked=self.reset_variable_state,
        )

        vertical_layout.addWidget(self.methods_container)
        vertical_layout.addStretch(2)
        vertical_layout.addWidget(self.reset_button)

        box.layout().addLayout(vertical_layout)

        self.variable_button_group = button_group

        gui.auto_apply(self.buttonsArea, self, "autocommit")

    def create_imputer(self, method, *args):
        # type: (Method, ...) -> impute.BaseImputeMethod
        if method == Method.Model:
            if self.learner is not None:
                return impute.Model(self.learner)
            else:
                return impute.Model(self.default_learner)
        elif method == Method.AsAboveSoBelow:
            assert self.default_method_index != Method.AsAboveSoBelow
            default = self.create_imputer(Method(self.default_method_index))
            m = AsDefault()
            m.method = default
            return m
        elif method == Method.Default and not args:  # global default values
            return impute.FixedValueByType(
                default_continuous=self.default_numeric_value,
                default_time=self.default_time
            )
        else:
            return METHODS[method](*args)

    @property
    def default_method_index(self):
        return self._default_method_index

    @default_method_index.setter
    def default_method_index(self, index):
        if self._default_method_index != index:
            assert index != Method.AsAboveSoBelow
            self._default_method_index = index
            self.default_button_group.button(index).setChecked(True)
            self.time_widget.setEnabled(index == Method.Default)
            self.numeric_value_widget.setEnabled(index == Method.Default)
            # update variable view
            self.update_varview()
            self._invalidate()

    def set_default_method(self, index):
        """Set the current selected default imputation method.
        """
        self.default_method_index = index

    def __on_default_numeric_value_edited(self):
        val = self.numeric_value_widget.value()
        if val != self.default_numeric_value:
            self.default_numeric_value = val
            if self.default_method_index == Method.Default:
                self._invalidate()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.cancel()
        self.closeContext()
        self.varmodel[:] = []
        self._variable_imputation_state = {}  # type: VariableState
        self.modified = False
        self.data = data

        if data is not None:
            self.varmodel[:] = data.domain.variables
            self.openContext(data.domain)
            # restore per variable imputation state
            self._restore_state(self._variable_imputation_state)
        self.reset_button.setEnabled(len(self.varmodel) > 0)

        self.update_varview()
        self.commit.now()

    @Inputs.learner
    def set_learner(self, learner):
        self.cancel()
        self.learner = learner or self.default_learner
        imputer = self.create_imputer(Method.Model)
        button = self.default_button_group.button(Method.Model)
        button.setText(imputer.name)

        variable_button = self.variable_button_group.button(Method.Model)
        variable_button.setText(imputer.name)

        if learner is not None:
            self.default_method_index = Method.Model

        self.update_varview()
        self.commit.deferred()

    def get_method_for_column(self, column_index):
        # type: (int) -> impute.BaseImputeMethod
        """
        Return the imputation method for column by its index.
        """
        assert 0 <= column_index < len(self.varmodel)
        idx = self.varmodel.index(column_index, 0)
        state = idx.data(StateRole)
        if state is None:
            state = (Method.AsAboveSoBelow, ())
        return self.create_imputer(state[0], *state[1])

    def _invalidate(self):
        self.modified = True
        if self.__task is not None:
            self.cancel()
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        self.cancel()
        self.warning()
        self.Error.imputation_failed.clear()
        self.Error.model_based_imputer_sparse.clear()

        if not self.data or not self.varmodel.rowCount():
            self.Outputs.data.send(self.data)
            self.modified = False
            return

        data = self.data
        impute_state = [
            (i, var, self.get_method_for_column(i))
            for i, var in enumerate(self.varmodel)
        ]
        # normalize to the effective method bypasing AsDefault
        impute_state = [
            (i, var, m.method if isinstance(m, AsDefault) else m)
            for i, var, m in impute_state
        ]

        def impute_one(method, var, data):
            # type: (impute.BaseImputeMethod, Variable, Table) -> Any
            # Readability counts, pylint: disable=no-else-raise
            if isinstance(method, impute.Model) and data.is_sparse():
                raise SparseNotSupported()
            elif isinstance(method, impute.DropInstances):
                return RowMask(method(data, var))
            elif not method.supports_variable(var):
                raise VariableNotSupported(var)
            else:
                return method(data, var)

        futures = []
        for _, var, method in impute_state:
            f = self.executor.submit(
                impute_one, copy.deepcopy(method), var, data)
            futures.append(f)

        w = qconcurrent.FutureSetWatcher(futures)
        w.doneAll.connect(self.__commit_finish)
        w.progressChanged.connect(self.__progress_changed)
        self.__task = Task(futures, w)
        self.progressBarInit()
        self.setInvalidated(True)

    @Slot()
    def __commit_finish(self):
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None
        futures = self.__task.futures
        assert len(futures) == len(self.varmodel)
        assert self.data is not None

        def get_variable(variable, future, drop_mask) \
                -> Optional[List[Orange.data.Variable]]:
            # Returns a (potentially empty) list of variables,
            # or None on failure that should interrupt the imputation
            assert future.done()
            try:
                res = future.result()
            except SparseNotSupported:
                self.Error.model_based_imputer_sparse()
                return []  # None?
            except VariableNotSupported:
                self.Warning.cant_handle_var(variable.name)
                return []
            except Exception:  # pylint: disable=broad-except
                log = logging.getLogger(__name__)
                log.info("Error for %s", variable.name, exc_info=True)
                self.Error.imputation_failed(variable.name)
                return None
            if isinstance(res, RowMask):
                drop_mask |= res.mask
                newvar = variable
            else:
                newvar = res
            if isinstance(newvar, Orange.data.Variable):
                newvar = [newvar]
            return newvar

        def create_data(attributes, class_vars):
            domain = Orange.data.Domain(
                attributes, class_vars, self.data.domain.metas)
            try:
                return self.data.from_table(domain, self.data[~drop_mask])
            except Exception:  # pylint: disable=broad-except
                log = logging.getLogger(__name__)
                log.info("Error", exc_info=True)
                self.Error.imputation_failed("Unknown")
                return None

        self.__task = None
        self.setInvalidated(False)
        self.progressBarFinished()

        attributes = []
        class_vars = []
        drop_mask = np.zeros(len(self.data), bool)
        for i, (var, fut) in enumerate(zip(self.varmodel, futures)):
            newvar = get_variable(var, fut, drop_mask)
            if newvar is None:
                data = None
                break
            if i < len(self.data.domain.attributes):
                attributes.extend(newvar)
            else:
                class_vars.extend(newvar)
        else:
            data = create_data(attributes, class_vars)

        self.Outputs.data.send(data)
        self.modified = False

    @Slot(int, int)
    def __progress_changed(self, n, d):
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None
        self.progressBarSet(100. * n / d)

    def cancel(self):
        self.__cancel(wait=False)

    def __cancel(self, wait=False):
        if self.__task is not None:
            task, self.__task = self.__task, None
            task.cancel()
            task.watcher.doneAll.disconnect(self.__commit_finish)
            task.watcher.progressChanged.disconnect(self.__progress_changed)
            if wait:
                concurrent.futures.wait(task.futures)
                task.watcher.flush()
            self.progressBarFinished()
            self.setInvalidated(False)

    def onDeleteWidget(self):
        self.__cancel(wait=True)
        super().onDeleteWidget()

    def send_report(self):
        specific = []
        for i, var in enumerate(self.varmodel):
            method = self.get_method_for_column(i)
            if not isinstance(method, AsDefault):
                specific.append("{} ({})".format(var.name, str(method)))

        default = self.create_imputer(Method.AsAboveSoBelow)
        if specific:
            self.report_items((
                ("Default method", default.name),
                ("Specific imputers", ", ".join(specific))
            ))
        else:
            self.report_items((("Method", default.name),))

    def _on_var_selection_changed(self):
        # Method is well documented, splitting it is not needed for readability,
        # thus pylint: disable=too-many-branches
        indexes = self.selection.selectedIndexes()
        self.methods_container.setEnabled(len(indexes) > 0)
        defmethod = (Method.AsAboveSoBelow, ())
        methods = [index.data(StateRole) for index in indexes]
        methods = [m if m is not None else defmethod for m in methods]
        methods = set(methods)
        selected_vars = [self.varmodel[index.row()] for index in indexes]
        has_discrete = any(var.is_discrete for var in selected_vars)
        fixed_value = None
        value_stack_enabled = False
        current_value_widget = None

        if len(methods) == 1:
            method_type, parameters = methods.pop()
            for m in Method:
                if method_type == m:
                    self.variable_button_group.button(m).setChecked(True)

            if method_type == Method.Default:
                (fixed_value,) = parameters

        elif self.variable_button_group.checkedButton() is not None:
            # Uncheck the current button
            self.variable_button_group.setExclusive(False)
            self.variable_button_group.checkedButton().setChecked(False)
            self.variable_button_group.setExclusive(True)
            assert self.variable_button_group.checkedButton() is None

        # Update variable methods GUI enabled state based on selection.
        for method in Method:
            # use a default constructed imputer to query support
            imputer = self.create_imputer(method)
            enabled = all(imputer.supports_variable(var) for var in
                          selected_vars)
            button = self.variable_button_group.button(method)
            button.setEnabled(enabled)

        # Update the "Value" edit GUI.
        if not has_discrete:
            # no discrete variables -> allow mass edit for all (continuous vars)
            value_stack_enabled = True
            current_value_widget = self.value_double
        elif len(selected_vars) == 1:
            # single discrete var -> enable and fill the values combo
            value_stack_enabled = True
            current_value_widget = self.value_combo
            self.value_combo.clear()
            self.value_combo.addItems(selected_vars[0].values)
        else:
            # mixed type selection -> disable
            value_stack_enabled = False
            current_value_widget = None
            self.variable_button_group.button(Method.Default).setEnabled(False)

        self.value_stack.setEnabled(value_stack_enabled)
        if current_value_widget is not None:
            self.value_stack.setCurrentWidget(current_value_widget)
            if fixed_value is not None:
                # set current value
                if current_value_widget is self.value_combo:
                    self.value_combo.setCurrentIndex(fixed_value)
                elif current_value_widget is self.value_double:
                    self.value_double.setValue(fixed_value)
                else:
                    assert False

    def set_method_for_current_selection(self, method_index):
        # type: (Method) -> None
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, method_index)

    def set_method_for_indexes(self, indexes, method_index):
        # type: (List[QModelIndex], Method) -> None
        if method_index == Method.AsAboveSoBelow:
            for index in indexes:
                self.varmodel.setData(index, None, StateRole)
        elif method_index == Method.Default:
            current = self.value_stack.currentWidget()
            if current is self.value_combo:
                value = self.value_combo.currentIndex()
            else:
                value = self.value_double.value()
            for index in indexes:
                state = (int(Method.Default), (value,))
                self.varmodel.setData(index, state, StateRole)
        else:
            state = (int(method_index), ())
            for index in indexes:
                self.varmodel.setData(index, state, StateRole)

        self.update_varview(indexes)
        self._invalidate()

    def update_varview(self, indexes=None):
        if indexes is None:
            indexes = map(self.varmodel.index, range(len(self.varmodel)))

        for index in indexes:
            self.varmodel.setData(
                index, self.get_method_for_column(index.row()),
                DisplayMethodRole)

    def _on_value_selected(self):
        # The fixed 'Value' in the widget has been changed by the user.
        self.variable_button_group.button(Method.Default).setChecked(True)
        self.set_method_for_current_selection(Method.Default)

    def reset_variable_state(self):
        indexes = list(map(self.varmodel.index, range(len(self.varmodel))))
        self.set_method_for_indexes(indexes, Method.AsAboveSoBelow)
        self.variable_button_group.button(Method.AsAboveSoBelow).setChecked(True)

    def _store_state(self):
        # type: () -> VariableState
        """
        Save the current variable imputation state
        """
        state = {}  # type: VariableState
        for i, var in enumerate(self.varmodel):
            index = self.varmodel.index(i)
            m = index.data(StateRole)
            if m is not None:
                state[var_key(var)] = m
        return state

    def _restore_state(self, state):
        # type: (VariableState) -> None
        """
        Restore the variable imputation state from the saved state
        """
        def check(state):
            # check if state is a proper State
            if isinstance(state, tuple) and len(state) == 2:
                m, p = state
                if isinstance(m, int) and isinstance(p, tuple) and \
                        0 <= m < len(Method):
                    return True
            return False

        for i, var in enumerate(self.varmodel):
            m = state.get(var_key(var), None)
            if check(m):
                self.varmodel.setData(self.varmodel.index(i), m, StateRole)

    def storeSpecificSettings(self):
        self._variable_imputation_state = self._store_state()
        super().storeSpecificSettings()


def __sample_data():  # pragma: no cover
    domain = Orange.data.Domain(
        [Orange.data.ContinuousVariable(f"c{i}") for i in range(3)]
        + [Orange.data.TimeVariable(f"t{i}") for i in range(3)],
        [])
    n = np.nan
    x = np.array([
        [1, 2, n, 1000, n, n],
        [2, n, 1, n, 2000, 2000]
    ])
    return Orange.data.Table(domain, x, np.empty((2, 0)))


if __name__ == "__main__":  # pragma: no cover
    # WidgetPreview(OWImpute).run(__sample_data())
    WidgetPreview(OWImpute).run(Orange.data.Table("brown-selected"))
