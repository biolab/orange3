from typing import Optional
from collections import namedtuple
from functools import singledispatch

import numpy as np

from AnyQt.QtCore import Qt, QSortFilterProxyModel, QSize
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QIcon
from AnyQt.QtWidgets import QLineEdit, QTableView, QSlider, QHeaderView, \
    QComboBox, QStyledItemDelegate, QWidget, QDateTimeEdit

from Orange.data import Variable, DiscreteVariable, ContinuousVariable, \
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


class VariableDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        self.parent().view.openPersistentEditor(index)
        super().paint(painter, option, index)

    def createEditor(self, parent, option, index):
        # TODO: check ordering (proxy)
        variable = index.data(VariableRole)
        values = index.data(ValuesRole)
        return _create_editor(variable, values, parent, self._commit_data)

    def _commit_data(self):
        self.commitData.emit(self.sender())

    def setEditorData(self, widget, index):
        _set_value(widget, index.model().data(index, ValueRole))

    def setModelData(self, widget, model, index):
        model.setData(index, _get_value(widget), ValueRole)


@singledispatch
def _create_editor(_: Variable, __: np.ndarray, ___: QWidget, cb) -> QWidget:
    raise NotImplementedError


@_create_editor.register(DiscreteVariable)
def _(variable: DiscreteVariable, _: np.ndarray, parent: QWidget, cb) -> QComboBox:
    combo = QComboBox(parent)
    combo.addItems(variable.values)
    combo.currentIndexChanged.connect(cb)
    return combo


@_create_editor.register(ContinuousVariable)
def _(_: ContinuousVariable, __: np.ndarray, parent: QWidget, cb) -> QSlider:
    slider = QSlider(parent, orientation=Qt.Horizontal)
    slider.valueChanged.connect(cb)
    return slider


@_create_editor.register(TimeVariable)
def _(variable: TimeVariable, _: np.ndarray, parent: QWidget, cb) -> QDateTimeEdit:
    edit = QDateTimeEdit(parent)
    date_format = "yyyy-MM-dd hh:mm:ss"
    if variable.have_date and not variable.have_time:
        date_format = "yyyy-MM-dd"
    elif not variable.have_date and variable.have_time:
        date_format = "hh:mm:ss"
    edit.setDisplayFormat(date_format)
    edit.dateChanged.connect(cb)
    return edit


@singledispatch
def _set_value(_: QWidget, __: float):
    raise NotImplementedError


@_set_value.register(QComboBox)
def _(widget: QComboBox, value: str):
    if isinstance(value, str):
        widget.setCurrentText(value)
    else:
        widget.setCurrentIndex(int(value))


@_set_value.register(QSlider)
def _(widget: QSlider, value: float):
    widget.setValue(int(value))


@_set_value.register(QDateTimeEdit)
def _(widget: QDateTimeEdit, value: float):
    # TODO
    widget.setDate(value)


@singledispatch
def _get_value(_: QWidget):
    raise NotImplementedError


@_get_value.register(QComboBox)
def _(widget: QComboBox):
    return widget.currentText()


@_get_value.register(QSlider)
def _(widget: QSlider):
    return widget.value()


@_get_value.register(QDateTimeEdit)
def _(widget: QDateTimeEdit):
    # TODO
    return widget.date()


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
        self.view = QTableView(sortingEnabled=True)
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
