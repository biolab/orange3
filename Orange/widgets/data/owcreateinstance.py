from typing import Optional, Callable, List, Union, Dict
from collections import namedtuple
from functools import singledispatch

import numpy as np

from AnyQt.QtCore import Qt, QSortFilterProxyModel, QSize, QDateTime, \
    QModelIndex, Signal, QPoint, QRect, QEvent
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QIcon, QPainter, \
    QColor
from AnyQt.QtWidgets import QLineEdit, QTableView, QSlider, \
    QComboBox, QStyledItemDelegate, QWidget, QDateTimeEdit, QHBoxLayout, \
    QDoubleSpinBox, QSizePolicy, QStyleOptionViewItem, QLabel, QMenu, QAction

from orangewidget.gui import Slider

from Orange.data import DiscreteVariable, ContinuousVariable, \
    TimeVariable, Table, StringVariable, Variable, Domain
from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

VariableRole = next(gui.OrangeUserRole)
ValuesRole = next(gui.OrangeUserRole)
ValueRole = next(gui.OrangeUserRole)


class VariableEditor(QWidget):
    valueChanged = Signal(float)

    def __init__(self, parent: QWidget, callback: Callable):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setAlignment(Qt.AlignLeft)
        self.setLayout(layout)
        self.valueChanged.connect(callback)

    @property
    def value(self) -> Union[int, float, str]:
        return NotImplemented

    @value.setter
    def value(self, value: Union[float, str]):
        raise NotImplementedError

    def sizeHint(self):
        return QSize(super().sizeHint().width(), 40)


class DiscreteVariableEditor(VariableEditor):
    valueChanged = Signal(int)

    def __init__(self, parent: QWidget, items: List[str], callback: Callable):
        super().__init__(parent, callback)
        self._combo = QComboBox(
            parent,
            maximumWidth=180,
            sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        )
        self._combo.addItems(items)
        self._combo.currentIndexChanged.connect(self.valueChanged)
        self.layout().addWidget(self._combo)

    @property
    def value(self) -> int:
        return self._combo.currentIndex()

    @value.setter
    def value(self, value: float):
        assert value == int(value)
        self._combo.setCurrentIndex(int(value))


class ContinuousVariableEditor(VariableEditor):
    MAX_FLOAT = 2147483647

    def __init__(self, parent: QWidget, variable: ContinuousVariable,
                 min_value: float, max_value: float, callback: Callable):
        super().__init__(parent, callback)

        if np.isnan(min_value) or np.isnan(max_value):
            raise ValueError("Min/Max cannot be NaN.")

        n_decimals = variable.number_of_decimals
        abs_max = max(abs(min_value), max_value)
        if abs_max * 10 ** n_decimals > self.MAX_FLOAT:
            n_decimals = int(np.log10(self.MAX_FLOAT / abs_max))

        self._value: float = min_value
        self._n_decimals: int = n_decimals
        self._min_value: float = self.__round_value(min_value)
        self._max_value: float = self.__round_value(max_value)

        sp_spin = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sp_spin.setHorizontalStretch(1)
        sp_slider = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sp_slider.setHorizontalStretch(5)
        sp_edit = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sp_edit.setHorizontalStretch(1)

        class DoubleSpinBox(QDoubleSpinBox):
            def sizeHint(self) -> QSize:
                size: QSize = super().sizeHint()
                return QSize(size.width(), size.height() + 2)

        self._spin = DoubleSpinBox(
            parent,
            value=self._min_value,
            minimum=-np.inf,
            maximum=np.inf,
            singleStep=10 ** (-self._n_decimals),
            decimals=self._n_decimals,
            minimumWidth=70,
            sizePolicy=sp_spin,
        )
        self._slider = Slider(
            parent,
            minimum=self.__map_to_slider(self._min_value),
            maximum=self.__map_to_slider(self._max_value),
            singleStep=1,
            orientation=Qt.Horizontal,
            sizePolicy=sp_slider,
        )
        self._label_min = QLabel(
            parent,
            text=variable.repr_val(min_value),
            alignment=Qt.AlignRight,
            minimumWidth=60,
            sizePolicy=sp_edit,
        )
        self._label_max = QLabel(
            parent,
            text=variable.repr_val(max_value),
            alignment=Qt.AlignLeft,
            minimumWidth=60,
            sizePolicy=sp_edit,
        )

        self._slider.valueChanged.connect(self._apply_slider_value)
        self._spin.valueChanged.connect(self._apply_spin_value)

        self.layout().addWidget(self._spin)
        self.layout().addWidget(self._label_min)
        self.layout().addWidget(self._slider)
        self.layout().addWidget(self._label_max)

        self.setFocusProxy(self._spin)

        def deselect():
            self._spin.lineEdit().deselect()
            try:
                self._spin.lineEdit().selectionChanged.disconnect(deselect)
            except TypeError:
                pass

        # Invoking self.setFocusProxy(self._spin), causes the
        # self._spin.lineEdit()s to have selected texts (focus is set to
        # provide keyboard functionality, i.e.: pressing ESC after changing
        # spinbox value). Since the spin text is selected only after the
        # delegate draws it, it cannot be deselected during initialization.
        # Therefore connect the deselect() function to
        # self._spin.lineEdit().selectionChanged only for editor creation.
        self._spin.lineEdit().selectionChanged.connect(deselect)

        self._slider.installEventFilter(self)
        self._spin.installEventFilter(self)

    @property
    def value(self) -> float:
        return self.__round_value(self._value)

    @value.setter
    def value(self, value: float):
        if self._value is None or self.__round_value(value) != self.value:
            self._value = value
            self.valueChanged.emit(self.value)
            self._spin.setValue(self.value)
            # prevent emitting self.valueChanged again, due to slider change
            slider_value = self.__map_to_slider(self.value)
            self._value = self.__map_from_slider(slider_value)
            self._slider.setValue(slider_value)
            self._value = value

    def _apply_slider_value(self):
        self.value = self.__map_from_slider(self._slider.value())

    def _apply_spin_value(self):
        self.value = self._spin.value()

    def __round_value(self, value):
        return round(value, self._n_decimals)

    def __map_to_slider(self, value: float) -> int:
        value = min(self._max_value, max(self._min_value, value))
        return round(value * 10 ** self._n_decimals)

    def __map_from_slider(self, value: int) -> float:
        return value * 10 ** (-self._n_decimals)

    def eventFilter(self, obj: Union[QSlider, QDoubleSpinBox], event: QEvent) \
            -> bool:
        if event.type() == QEvent.Wheel:
            return True
        return super().eventFilter(obj, event)


class StringVariableEditor(VariableEditor):
    valueChanged = Signal()

    def __init__(self, parent: QWidget, callback: Callable):
        super().__init__(parent, callback)
        self._edit = QLineEdit(
            parent,
            sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        )
        self._edit.textChanged.connect(self.valueChanged)
        self.layout().addWidget(self._edit)
        self.setFocusProxy(self._edit)

    @property
    def value(self) -> str:
        return self._edit.text()

    @value.setter
    def value(self, value: str):
        self._edit.setText(value)


class TimeVariableEditor(VariableEditor):
    DATE_FORMAT = "yyyy-MM-dd"
    TIME_FORMAT = "hh:mm:ss"

    def __init__(self, parent: QWidget, variable: TimeVariable,
                 callback: Callable):
        super().__init__(parent, callback)
        self._value: float = 0
        self._variable: TimeVariable = variable

        if variable.have_date and not variable.have_time:
            self._format = TimeVariableEditor.DATE_FORMAT
        elif not variable.have_date and variable.have_time:
            self._format = TimeVariableEditor.TIME_FORMAT
        else:
            self._format = f"{TimeVariableEditor.DATE_FORMAT} " \
                           f"{TimeVariableEditor.TIME_FORMAT}"

        class DateTimeEdit(QDateTimeEdit):
            def sizeHint(self) -> QSize:
                size: QSize = super().sizeHint()
                return QSize(size.width(), size.height() + 2)

        self._edit = DateTimeEdit(
            parent,
            dateTime=self.__map_to_datetime(self._value),
            displayFormat=self._format,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        )
        self._edit.dateTimeChanged.connect(self._apply_edit_value)

        self.layout().addWidget(self._edit)
        self.setFocusProxy(self._edit)

        self._edit.installEventFilter(self)

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        if value != self.value:
            self._value = value
            self.valueChanged.emit(self.value)
            self._edit.setDateTime(self.__map_to_datetime(self.value))

    def _apply_edit_value(self):
        self.value = self.__map_from_datetime(self._edit.dateTime())

    def __map_from_datetime(self, date_time: QDateTime) -> float:
        return self._variable.to_val(date_time.toString(self._format))

    def __map_to_datetime(self, value: float) -> QDateTime:
        return QDateTime.fromString(self._variable.repr_val(value),
                                    self._format)

    def eventFilter(self, obj: QDateTimeEdit, event: QEvent) -> bool:
        if event.type() == QEvent.Wheel:
            return True
        return super().eventFilter(obj, event)


class VariableDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex):
        self.parent().view.openPersistentEditor(index)
        super().paint(painter, option, index)

    def createEditor(self, parent: QWidget, _: QStyleOptionViewItem,
                     index: QModelIndex) -> VariableEditor:
        variable = index.data(VariableRole)
        values = index.data(ValuesRole)
        return _create_editor(variable, values, parent, self._commit_data)

    def _commit_data(self):
        editor = self.sender()
        assert isinstance(editor, VariableEditor)
        self.commitData.emit(editor)

    # pylint: disable=no-self-use
    def setEditorData(self, editor: VariableEditor, index: QModelIndex):
        editor.value = index.model().data(index, ValueRole)

    # pylint: disable=no-self-use
    def setModelData(self, editor: VariableEditor,
                     model: QSortFilterProxyModel, index: QModelIndex):
        model.setData(index, editor.value, ValueRole)

    # pylint: disable=no-self-use
    def updateEditorGeometry(self, editor: VariableEditor,
                             option: QStyleOptionViewItem, _: QModelIndex):
        rect: QRect = option.rect
        if isinstance(editor, ContinuousVariableEditor):
            width = editor.sizeHint().width()
            if width > rect.width():
                rect.setWidth(width)
        editor.setGeometry(rect)

    # pylint: disable=no-self-use
    def sizeHint(self, _: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return _create_editor(index.data(role=VariableRole), np.array([0]),
                              None, lambda: 1).sizeHint()


@singledispatch
def _create_editor(*_) -> VariableEditor:
    raise NotImplementedError


@_create_editor.register(DiscreteVariable)
def _(variable: DiscreteVariable, _: np.ndarray,
      parent: QWidget, callback: Callable) -> DiscreteVariableEditor:
    return DiscreteVariableEditor(parent, variable.values, callback)


@_create_editor.register(ContinuousVariable)
def _(variable: ContinuousVariable, values: np.ndarray,
      parent: QWidget, callback: Callable) -> ContinuousVariableEditor:
    return ContinuousVariableEditor(parent, variable, np.nanmin(values),
                                    np.nanmax(values), callback)


@_create_editor.register(StringVariable)
def _(_: StringVariable, __: np.ndarray, parent: QWidget,
      callback: Callable) -> StringVariableEditor:
    return StringVariableEditor(parent, callback)


@_create_editor.register(TimeVariable)
def _(variable: TimeVariable, _: np.ndarray,
      parent: QWidget, callback: Callable) -> TimeVariableEditor:
    return TimeVariableEditor(parent, variable, callback)


def majority(values: np.ndarray) -> int:
    return np.bincount(values[~np.isnan(values)].astype(int)).argmax()


def disc_random(values: np.ndarray) -> int:
    return np.random.randint(low=np.nanmin(values), high=np.nanmax(values) + 1)


def cont_random(values: np.ndarray) -> float:
    return np.random.uniform(low=np.nanmin(values), high=np.nanmax(values))


class VariableItemModel(QStandardItemModel):
    dataHasNanColumn = Signal()

    # pylint: disable=dangerous-default-value
    def set_data(self, data: Table, saved_values={}):
        domain = data.domain
        variables = [(TableModel.Attribute, a) for a in domain.attributes] + \
                    [(TableModel.ClassVar, c) for c in domain.class_vars] + \
                    [(TableModel.Meta, m) for m in domain.metas]
        for place, variable in variables:
            if variable.is_primitive():
                values = data.get_column_view(variable)[0].astype(float)
                if all(np.isnan(values)):
                    self.dataHasNanColumn.emit()
                    continue
            else:
                values = np.array([])
            color = TableModel.ColorForRole.get(place)
            self._add_row(variable, values, color,
                          saved_values.get(variable.name))

    def _add_row(self, variable: Variable, values: np.ndarray, color: QColor,
                 saved_value: Optional[Union[int, float, str]]):
        var_item = QStandardItem()
        var_item.setData(variable.name, Qt.DisplayRole)
        var_item.setToolTip(variable.name)
        var_item.setIcon(self._variable_icon(variable))
        var_item.setEditable(False)
        if color:
            var_item.setBackground(color)

        control_item = QStandardItem()
        control_item.setData(variable, VariableRole)
        control_item.setData(values, ValuesRole)
        if color:
            control_item.setBackground(color)

        value = self._default_for_variable(variable, values)
        if saved_value is not None and not \
                (variable.is_discrete and saved_value >= len(variable.values)):
            value = saved_value
        control_item.setData(value, ValueRole)

        self.appendRow([var_item, control_item])

    @staticmethod
    def _default_for_variable(variable: Variable, values: np.ndarray) \
            -> Union[float, int, str]:
        if variable.is_continuous:
            return round(np.nanmedian(values), variable.number_of_decimals)
        elif variable.is_discrete:
            return majority(values)
        elif variable.is_string:
            return ""
        else:
            raise NotImplementedError

    @staticmethod
    def _variable_icon(variable: Variable) -> QIcon:
        if variable.is_discrete:
            return gui.attributeIconDict[1]
        elif variable.is_time:
            return gui.attributeIconDict[4]
        elif variable.is_continuous:
            return gui.attributeIconDict[2]
        elif variable.is_string:
            return gui.attributeIconDict[3]
        else:
            return gui.attributeIconDict[-1]


class OWCreateInstance(OWWidget):
    name = "Create Instance"
    description = "Interactively create a data instance from sample dataset."
    icon = "icons/CreateInstance.svg"
    category = "Transform"
    keywords = ["simulator"]
    priority = 2310

    class Inputs:
        data = Input("Data", Table)
        reference = Input("Reference", Table)

    class Outputs:
        data = Output("Data", Table)

    class Information(OWWidget.Information):
        nans_removed = Msg("Variables with only missing values were "
                           "removed from the list.")

    want_main_area = False
    ACTIONS = ["median", "mean", "random", "input"]
    HEADER = [["name", "Variable"],
              ["variable", "Value"]]
    Header = namedtuple(
        "header", [tag for tag, _ in HEADER]
    )(*range(len(HEADER)))

    values: Dict[str, Union[float, str]] = Setting({}, schema_only=True)
    append_to_data = Setting(True)
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.reference: Optional[Table] = None

        self.filter_edit = QLineEdit(textChanged=self.__filter_edit_changed,
                                     placeholderText="Filter...")
        self.view = QTableView(sortingEnabled=True,
                               contextMenuPolicy=Qt.CustomContextMenu,
                               selectionMode=QTableView.NoSelection)
        self.view.customContextMenuRequested.connect(self.__menu_requested)
        self.view.setItemDelegateForColumn(
            self.Header.variable, VariableDelegate(self)
        )
        self.view.verticalHeader().hide()
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.horizontalHeader().setMaximumSectionSize(350)

        self.model = VariableItemModel(self)
        self.model.setHorizontalHeaderLabels([x for _, x in self.HEADER])
        self.model.dataChanged.connect(self.__table_data_changed)
        self.model.dataHasNanColumn.connect(self.Information.nans_removed)
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setFilterKeyColumn(-1)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setSourceModel(self.model)
        self.view.setModel(self.proxy_model)

        vbox = gui.vBox(self.controlArea, box=True)
        vbox.layout().addWidget(self.filter_edit)
        vbox.layout().addWidget(self.view)

        box = gui.hBox(vbox, objectName="buttonBox")
        gui.rubber(box)
        for name in self.ACTIONS:
            gui.button(
                box, self, name.capitalize(),
                lambda *args, fun=name: self._initialize_values(fun),
                autoDefault=False
            )
        gui.rubber(box)

        gui.checkBox(self.buttonsArea, self, "append_to_data",
                     "Append this instance to input data",
                     callback=self.commit.deferred)
        gui.rubber(self.buttonsArea)
        gui.auto_apply(self.buttonsArea, self, "auto_commit")

        self.settingsAboutToBePacked.connect(self.pack_settings)

    def __filter_edit_changed(self):
        self.proxy_model.setFilterFixedString(self.filter_edit.text().strip())

    def __table_data_changed(self):
        self.commit.deferred()

    def __menu_requested(self, point: QPoint):
        index = self.view.indexAt(point)
        model: QSortFilterProxyModel = index.model()
        source_index = model.mapToSource(index)
        menu = QMenu(self)
        for action in self._create_actions(source_index):
            menu.addAction(action)
        menu.popup(self.view.viewport().mapToGlobal(point))

    def _create_actions(self, index: QModelIndex) -> List[QAction]:
        actions = []
        for name in self.ACTIONS:
            action = QAction(name.capitalize(), self)
            action.triggered.connect(
                lambda *args, fun=name: self._initialize_values(fun, [index])
            )
            actions.append(action)
        return actions

    def _initialize_values(self, fun: str, indices: List[QModelIndex] = None):
        cont_fun = {"median": np.nanmedian,
                    "mean": np.nanmean,
                    "random": cont_random,
                    "input": np.nanmean}.get(fun, NotImplemented)
        disc_fun = {"median": majority,
                    "mean": majority,
                    "random": disc_random,
                    "input": majority}.get(fun, NotImplemented)

        if not self.data or fun == "input" and not self.reference:
            return

        self.model.dataChanged.disconnect(self.__table_data_changed)
        rows = range(self.proxy_model.rowCount()) if indices is None else \
            [index.row() for index in indices]
        for row in rows:
            index = self.model.index(row, self.Header.variable)
            variable = self.model.data(index, VariableRole)

            if fun == "input":
                if variable not in self.reference.domain:
                    continue
                values = self.reference.get_column_view(variable)[0]
                if variable.is_primitive():
                    values = values.astype(float)
                    if all(np.isnan(values)):
                        continue
            else:
                values = self.model.data(index, ValuesRole)

            if variable.is_continuous:
                value = cont_fun(values)
                value = round(value, variable.number_of_decimals)
            elif variable.is_discrete:
                value = disc_fun(values)
            elif variable.is_string:
                value = ""
            else:
                raise NotImplementedError

            self.model.setData(index, value, ValueRole)
        self.model.dataChanged.connect(self.__table_data_changed)
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data: Table):
        self.data = data
        self._set_model_data()
        self.commit.now()

    def _set_model_data(self):
        self.Information.nans_removed.clear()
        self.model.removeRows(0, self.model.rowCount())
        if not self.data:
            return

        self.model.set_data(self.data, self.values)
        self.values = {}
        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.resizeColumnsToContents()
        self.view.resizeRowsToContents()
        self.view.horizontalHeader().setStretchLastSection(True)

    @Inputs.reference
    def set_reference(self, data: Table):
        self.reference = data

    @gui.deferred
    def commit(self):
        output_data = None
        if self.data:
            output_data = self._create_data_from_values()
            if self.append_to_data:
                output_data = self._append_to_data(output_data)
        self.Outputs.data.send(output_data)

    def _create_data_from_values(self) -> Table:
        data = Table.from_domain(self.data.domain, 1)
        with data.unlocked():
            data.name = "created"
            if data.X.size:
                data.X[:] = np.nan
            if data.Y.size:
                data.Y[:] = np.nan
            for i, m in enumerate(self.data.domain.metas):
                data.metas[:, i] = "" if m.is_string else np.nan

            values = self._get_values()
            for var_name, value in values.items():
                data[:, var_name] = value
        return data

    def _append_to_data(self, data: Table) -> Table:
        assert self.data
        assert len(data) == 1

        var = DiscreteVariable("Source ID", values=(self.data.name, data.name))
        data = Table.concatenate([self.data, data], axis=0)
        domain = Domain(data.domain.attributes, data.domain.class_vars,
                        data.domain.metas + (var,))
        data = data.transform(domain)
        with data.unlocked(data.metas):
            data.metas[: len(self.data), -1] = 0
            data.metas[len(self.data):, -1] = 1
        return data

    def _get_values(self) -> Dict[str, Union[str, float]]:
        values = {}
        for row in range(self.model.rowCount()):
            index = self.model.index(row, self.Header.variable)
            values[self.model.data(index, VariableRole).name] = \
                self.model.data(index, ValueRole)
        return values

    def send_report(self):
        if not self.data:
            return
        self.report_domain("Input", self.data.domain)
        self.report_domain("Output", self.data.domain)
        items = []
        values: Dict = self._get_values()
        for var in self.data.domain.variables + self.data.domain.metas:
            val = values.get(var.name, np.nan)
            if var.is_primitive():
                val = var.repr_val(val)
            items.append([f"{var.name}:", val])
        self.report_table("Values", items)

    @staticmethod
    def sizeHint():
        return QSize(600, 500)

    def pack_settings(self):
        self.values: Dict[str, Union[str, float]] = self._get_values()


if __name__ == "__main__":  # pragma: no cover
    table = Table("housing")
    WidgetPreview(OWCreateInstance).run(set_data=table,
                                        set_reference=table[:1])
