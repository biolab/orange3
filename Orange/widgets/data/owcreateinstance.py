from typing import Optional, Callable, List, Union
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
    TimeVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.state_summary import format_summary_details, \
    format_multiple_summaries
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output

VariableRole = next(gui.OrangeUserRole)
ValuesRole = next(gui.OrangeUserRole)
ValueRole = next(gui.OrangeUserRole)


class VariableEditor(QWidget):
    value_changed = Signal(float)

    def __init__(self, parent: QWidget, callback: Callable):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
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

        sp_spin = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sp_slider = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sp_slider.setHorizontalStretch(5)
        sp_edit = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sp_edit.setHorizontalStretch(1)

        self._spin = QDoubleSpinBox(
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
        self.layout().setContentsMargins(4, 0, 0, 4)
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


# TODO
class TimeVariableEditor(VariableEditor):
    def __init__(self, parent: QWidget, have_date: bool,
                 have_time: bool, callback: Callable):
        super().__init__(parent)
        self._editor = QDateTimeEdit(parent)
        date_format = "yyyy-MM-dd hh:mm:ss"
        if have_date and not have_time:
            date_format = "yyyy-MM-dd"
        elif not have_date and have_time:
            date_format = "hh:mm:ss"
        self._editor.setDisplayFormat(date_format)
        self._editor.dateChanged.connect(callback)
        self.layout().addWidget(self._editor)

    @property
    def value(self):
        return self._editor.date()

    @value.setter
    def value(self, value):
        self._editor.setDateTime(QDateTime(value))


class VariableDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex):
        self.parent().view.openPersistentEditor(index)
        super().paint(painter, option, index)

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem,
                     index: QModelIndex) -> VariableEditor:
        variable = index.data(VariableRole)
        values = index.data(ValuesRole)
        return _create_editor(variable, values, parent, self._commit_data)

    def _commit_data(self):
        editor = self.sender()
        assert isinstance(editor, VariableEditor)
        self.commitData.emit(editor)

    def setEditorData(self, editor: VariableEditor, index: QModelIndex):
        editor.value = index.model().data(index, ValueRole)

    def setModelData(self, editor: VariableEditor,
                     model: QSortFilterProxyModel, index: QModelIndex):
        model.setData(index, editor.value, ValueRole)

    def sizeHint(self, option: QStyleOptionViewItem,
                 index: QModelIndex) -> QSize:
        sh = super().sizeHint(option, index)
        return QSize(sh.width(), sh.height() + 20)


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


@_create_editor.register(TimeVariable)
def _(variable: TimeVariable, _: np.ndarray,
      parent: QWidget, callback: Callable) -> TimeVariableEditor:
    return TimeVariableEditor(parent, variable.have_date,
                              variable.have_time, callback)


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

    want_main_area = True
    HEADER = [
        ['name', "Variable"],
        ['variable', "Value"],
    ]

    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.reference: Optional[Table] = None
        self.Header = \
            namedtuple("header", [tag for tag, _ in self.HEADER])(
                *range(len(self.HEADER))
            )
        self.setup_gui()

    def setup_gui(self):
        self._add_controls()
        self._add_table()
        self._set_input_summary()
        self._set_output_summary()

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Initialize values")
        kwargs = {"autoDefault": False}
        gui.button(box, self, "Median", self.__median_button_clicked, **kwargs)
        gui.button(box, self, "Mean", self.__mean_button_clicked, **kwargs)
        gui.button(box, self, "Input", self.__input_button_clicked, **kwargs)
        gui.rubber(self.controlArea)
        gui.auto_apply(self.controlArea, self, "auto_commit")

    def __median_button_clicked(self):
        print("median")
        self.commit()

    def __mean_button_clicked(self):
        print("mean")
        self.commit()

    def __input_button_clicked(self):
        print("input")
        self.commit()

    def _add_table(self):
        self.model = QStandardItemModel(self)
        self.model.dataChanged.connect(self.__table_data_changed)
        self.filter_edit = QLineEdit(
            textChanged=self.__filter_edit_changed,
            placeholderText="Filter..."
        )
        self.view = QTableView(sortingEnabled=True,
                               selectionMode=QTableView.NoSelection)
        self.view.setItemDelegateForColumn(
            self.Header.variable, VariableDelegate(self)
        )
        self.view.verticalHeader().hide()
        header: QHeaderView = self.view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setMaximumSectionSize(300)
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setFilterKeyColumn(-1)
        self.proxy_model.setFilterCaseSensitivity(False)
        self.proxy_model.setSourceModel(self.model)
        self.view.setModel(self.proxy_model)
        self.mainArea.layout().addWidget(self.filter_edit)
        self.mainArea.layout().addWidget(self.view)

    def __table_data_changed(self):
        self.commit()

    def __filter_edit_changed(self):
        self.proxy_model.setFilterFixedString(self.filter_edit.text().strip())

    @Inputs.data
    def set_data(self, data: Table):
        self.data = data
        self._set_model_data()

    def _set_model_data(self):
        def variable_icon() -> QIcon:
            if var.is_discrete:
                return gui.attributeIconDict[1]
            elif var.is_time:
                return gui.attributeIconDict[4]
            elif var.is_continuous:
                return gui.attributeIconDict[2]
            else:
                return gui.attributeIconDict[-1]

        def add_row():
            var_item = QStandardItem()
            var_item.setData(var.name, Qt.DisplayRole)
            var_item.setIcon(variable_icon())
            var_item.setEditable(False)

            control_item = QStandardItem()
            values = self.data.get_column_view(var)[0]
            control_item.setData(var, VariableRole)
            control_item.setData(values, ValuesRole)
            value = np.nanmedian(values.astype(float))
            if var.is_continuous:
                value = round(value, var.number_of_decimals)
            control_item.setData(value, ValueRole)

            self.model.appendRow([var_item, control_item])

        self.model.clear()
        self.model.setHorizontalHeaderLabels([x for _, x in self.HEADER])
        if not self.data:
            return
        for var in self.data.domain.variables + self.data.domain.metas:
            if var.is_primitive():
                add_row()
        self.view.resizeColumnsToContents()
        self.view.resizeRowsToContents()

    @Inputs.reference
    def set_reference(self, data: Table):
        self.reference = data

    def handleNewSignals(self):
        self._set_input_summary()
        self.unconditional_commit()

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
            data = self._set_sample_values(data)
        self._set_output_summary(data)
        self.Outputs.data.send(data)

    def _set_sample_values(self, data: Table) -> Table:
        for row in range(self.model.rowCount()):
            model: QStandardItemModel = self.model
            index = model.index(row, self.Header.variable)
            var = model.data(index, VariableRole)
            data[:, var] = model.data(index, ValueRole)
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
