from typing import Optional, Callable, List, Union, Dict
from collections import namedtuple
from functools import singledispatch

import numpy as np

from AnyQt.QtCore import Qt, QSortFilterProxyModel, QSize, QDateTime, \
    QModelIndex, Signal
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QIcon, QPainter
from AnyQt.QtWidgets import QLineEdit, QTableView, QSlider, QHeaderView, \
    QComboBox, QStyledItemDelegate, QWidget, QDateTimeEdit, QHBoxLayout, \
    QDoubleSpinBox, QSizePolicy, QStyleOptionViewItem, QLabel

from Orange.data import DiscreteVariable, ContinuousVariable, \
    TimeVariable, Table, StringVariable, Variable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.state_summary import format_summary_details, \
    format_multiple_summaries
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

VariableRole = next(gui.OrangeUserRole)
ValuesRole = next(gui.OrangeUserRole)
ValueRole = next(gui.OrangeUserRole)


class VariableEditor(QWidget):
    value_changed = Signal(float)

    def __init__(self, parent: QWidget, callback: Callable):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        self.setLayout(layout)
        self.value_changed.connect(callback)

    @property
    def value(self) -> Union[int, float]:
        return NotImplemented

    @value.setter
    def value(self, value: float):
        raise NotImplementedError


class DiscreteVariableEditor(VariableEditor):
    value_changed = Signal(int)

    def __init__(self, parent: QWidget, items: List[str], callback: Callable):
        super().__init__(parent, callback)
        self._combo = QComboBox(parent)
        self._combo.addItems(items)
        self._combo.currentIndexChanged.connect(self.value_changed)
        self.layout().addWidget(self._combo)
        self.layout().setContentsMargins(0, 1, 0, 0)

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
        sp_slider.setHorizontalStretch(6)
        sp_edit = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sp_edit.setHorizontalStretch(2)

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
        self._slider = QSlider(
            parent,
            minimum=self.__map_to_slider(self._min_value),
            maximum=self.__map_to_slider(self._max_value),
            singleStep=1,
            orientation=Qt.Horizontal,
            minimumWidth=20,
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
        self.setMinimumWidth(200)

        self.setFocusProxy(self._spin)

        # FIXME: after setting focus proxy to the spin, the text is highlighted

        def deselect():
            self._spin.lineEdit().deselect()
            try:
                self._spin.lineEdit().selectionChanged.disconnect(deselect)
            except TypeError:
                pass

        self._spin.lineEdit().selectionChanged.connect(deselect)

    @property
    def value(self) -> float:
        return self.__round_value(self._value)

    @value.setter
    def value(self, value: float):
        if self._value is None or self.__round_value(value) != self.value:
            self._value = value
            self.value_changed.emit(self.value)
            self._spin.setValue(self.value)
            self._slider.setValue(self.__map_to_slider(self.value))

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


class StringVariableEditor(VariableEditor):
    value_changed = Signal()

    def __init__(self, parent: QWidget, callback: Callable):
        super().__init__(parent, callback)
        self._edit = QLineEdit(parent)
        self._edit.textChanged.connect(self.value_changed)
        self._edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout().addWidget(self._edit)
        self.layout().setContentsMargins(5, 0, 5, 0)
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
        )
        self._edit.dateTimeChanged.connect(self._apply_edit_value)

        self.layout().addWidget(self._edit)
        self.setFocusProxy(self._edit)

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        if value != self.value:
            self._value = value
            self.value_changed.emit(self.value)
            self._edit.setDateTime(self.__map_to_datetime(self.value))

    def _apply_edit_value(self):
        self.value = self.__map_from_datetime(self._edit.dateTime())

    def __map_from_datetime(self, date_time: QDateTime) -> float:
        return self._variable.to_val(date_time.toString(self._format))

    def __map_to_datetime(self, value: float) -> QDateTime:
        return QDateTime.fromString(self._variable.repr_val(value),
                                    self._format)


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

    @staticmethod
    def setEditorData(editor: VariableEditor, index: QModelIndex):
        editor.value = index.model().data(index, ValueRole)

    @staticmethod
    def setModelData(editor: VariableEditor, model: QSortFilterProxyModel,
                     index: QModelIndex):
        model.setData(index, editor.value, ValueRole)

    def sizeHint(self, option: QStyleOptionViewItem,
                 index: QModelIndex) -> QSize:
        sh = super().sizeHint(option, index)
        return QSize(sh.width(), 40)


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

    def set_data(self, data: Table, saved_values={}):
        for variable in data.domain.variables + data.domain.metas:
            if variable.is_primitive():
                values = data.get_column_view(variable)[0].astype(float)
                if all(np.isnan(values)):
                    self.dataHasNanColumn.emit()
                    continue
            else:
                values = np.array([])
            self._add_row(variable, values, saved_values.get(variable.name))

    def _add_row(self, variable: Variable, values: np.ndarray,
                 saved_value: Optional[Union[int, float, str]]):
        var_item = QStandardItem()
        var_item.setData(variable.name, Qt.DisplayRole)
        var_item.setIcon(self._variable_icon(variable))
        var_item.setEditable(False)

        control_item = QStandardItem()
        control_item.setData(variable, VariableRole)
        control_item.setData(values, ValuesRole)

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
    category = "Data"
    keywords = ["simulator"]
    priority = 4000

    class Inputs:
        data = Input("Data", Table)
        reference = Input("Reference", Table)

    class Outputs:
        data = Output("Data", Table)

    class Information(OWWidget.Information):
        nans_removed = Msg("Variables with only missing values were "
                           "removed from the list.")

    want_main_area = False
    HEADER = [
        ['name', "Variable"],
        ['variable', "Value"],
    ]
    Header = namedtuple(
        "header", [tag for tag, _ in HEADER]
    )(*range(len(HEADER)))

    values = Setting(
        {}, schema_only=True)  # type: Dict[str, Union[float, str]]
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.reference: Optional[Table] = None
        self.__pending_values: Dict[str, Union[float, str]] = self.values

        self.filter_edit = QLineEdit(textChanged=self.__filter_edit_changed,
                                     placeholderText="Filter...")
        self.view = QTableView(sortingEnabled=True,
                               selectionMode=QTableView.NoSelection)
        self.view.setItemDelegateForColumn(
            self.Header.variable, VariableDelegate(self)
        )
        self.view.verticalHeader().hide()
        header: QHeaderView = self.view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setMaximumSectionSize(300)

        self.model = VariableItemModel(self)
        self.model.dataChanged.connect(self.__table_data_changed)
        self.model.dataHasNanColumn.connect(self.Information.nans_removed)
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setFilterKeyColumn(-1)
        self.proxy_model.setFilterCaseSensitivity(False)
        self.proxy_model.setSourceModel(self.model)
        self.view.setModel(self.proxy_model)

        vbox = gui.vBox(self.controlArea, box=True)
        vbox.layout().addWidget(self.filter_edit)
        vbox.layout().addWidget(self.view)

        box = gui.hBox(vbox)
        gui.rubber(box)
        kwargs = {"autoDefault": False}
        gui.button(box, self, "Median", self.__median_button_clicked, **kwargs)
        gui.button(box, self, "Mean", self.__mean_button_clicked, **kwargs)
        gui.button(box, self, "Random", self.__random_button_clicked, **kwargs)
        gui.button(box, self, "Input", self.__input_button_clicked, **kwargs)
        gui.rubber(box)

        box = gui.auto_apply(self.controlArea, self, "auto_commit")
        box.button.setFixedWidth(180)
        box.layout().insertStretch(0)

        self._set_input_summary()
        self._set_output_summary()

    def __filter_edit_changed(self):
        self.proxy_model.setFilterFixedString(self.filter_edit.text().strip())

    def __table_data_changed(self):
        self.commit()

    def __median_button_clicked(self):
        self._initialize_values("median")

    def __mean_button_clicked(self):
        self._initialize_values("mean")

    def __random_button_clicked(self):
        self._initialize_values("random")

    def __input_button_clicked(self):
        if not self.reference:
            return
        self._initialize_values("input")

    def _initialize_values(self, fun: str):
        cont_fun = {"median": np.nanmedian,
                    "mean": np.nanmean,
                    "random": cont_random,
                    "input": np.nanmean}.get(fun, NotImplemented)
        disc_fun = {"median": majority,
                    "mean": majority,
                    "random": disc_random,
                    "input": majority}.get(fun, NotImplemented)

        if not self.data:
            return

        self.model.dataChanged.disconnect(self.__table_data_changed)
        for row in range(self.model.rowCount()):
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
        self.commit()

    @Inputs.data
    def set_data(self, data: Table):
        self.data = data
        self._set_input_summary()
        self._clear()
        self._set_model_data()
        self.unconditional_commit()

    def _clear(self):
        self.Information.nans_removed.clear()
        self.model.clear()
        self.model.setHorizontalHeaderLabels([x for _, x in self.HEADER])

    def _set_model_data(self):
        if not self.data:
            return

        self.model.set_data(self.data, self.__pending_values)
        self.__pending_values = {}
        self.view.resizeColumnsToContents()
        self.view.resizeRowsToContents()

    @Inputs.reference
    def set_reference(self, data: Table):
        self.reference = data
        self._set_input_summary()

    def _set_input_summary(self):
        n_data = len(self.data) if self.data else 0
        n_refs = len(self.reference) if self.reference else 0
        summary, details, kwargs = self.info.NoInput, "", {}

        if self.data or self.reference:
            summary = f"{self.info.format_number(n_data)}, " \
                      f"{self.info.format_number(n_refs)}"
            data_list = [("Data", self.data), ("Reference", self.reference)]
            details = format_multiple_summaries(data_list)
            kwargs = {"format": Qt.RichText}
        self.info.set_input_summary(summary, details, **kwargs)

    def _set_output_summary(self, data: Optional[Table] = None):
        if data:
            summary, details = len(data), format_summary_details(data)
        else:
            summary, details = self.info.NoOutput, ""
        self.info.set_output_summary(summary, details)

    def commit(self):
        data = None
        if self.data:
            data = Table.from_domain(self.data.domain, 1)
            data.X[:] = np.nan
            data.Y[:] = np.nan
            for i, m in enumerate(self.data.domain.metas):
                data.metas[:, i] = "" if m.is_string else np.nan
            data = self._set_values(data)
        self._set_output_summary(data)
        self.Outputs.data.send(data)

    def _set_values(self, data: Table) -> Table:
        self.values = {}
        for row in range(self.model.rowCount()):
            model: QStandardItemModel = self.model
            index = model.index(row, self.Header.variable)
            var = model.data(index, VariableRole)
            value = model.data(index, ValueRole)
            data[:, var] = value
            self.values[model.data(index, VariableRole).name] = value
        return data

    def send_report(self):
        pass

    @staticmethod
    def sizeHint():
        return QSize(800, 500)


if __name__ == "__main__":  # pragma: no cover
    table = Table("heart_disease")
    WidgetPreview(OWCreateInstance).run(set_data=table,
                                        set_reference=table[::2])
