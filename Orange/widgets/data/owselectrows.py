import enum
from collections import OrderedDict
from datetime import datetime, timezone, timedelta

import numpy as np

from AnyQt.QtWidgets import (
    QWidget, QTableWidget, QHeaderView, QComboBox, QLineEdit, QToolButton,
    QMessageBox, QMenu, QListView, QGridLayout, QPushButton, QSizePolicy,
    QLabel, QDateTimeEdit)
from AnyQt.QtGui import (QDoubleValidator, QStandardItemModel, QStandardItem,
                         QFontMetrics, QPalette)
from AnyQt.QtCore import Qt, QPoint, QPersistentModelIndex, QLocale, \
    QDateTime, QDate, QTime

from orangewidget.utils.combobox import ComboBoxSearch
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable, Table)
import Orange.data.filter as data_filter
from Orange.data.filter import FilterContinuous, FilterString
from Orange.data.sql.table import SqlTable
from Orange.preprocess import Remove
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils import vartype
from Orange.widgets import report
from Orange.widgets.widget import Msg
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)


class SelectRowsContextHandler(DomainContextHandler):
    """Context handler that filters conditions"""

    # pylint: disable=arguments-differ
    def is_valid_item(self, setting, condition, attrs, metas):
        """Return True if condition applies to a variable in given domain."""
        varname, *_ = condition
        return varname in attrs or varname in metas

    def encode_setting(self, context, setting, value):
        if setting.name != 'conditions':
            return super().encode_settings(context, setting, value)

        encoded = []
        CONTINUOUS = vartype(ContinuousVariable("x"))
        for attr, op, values in value:
            if isinstance(attr, str):
                if OWSelectRows.AllTypes.get(attr) == CONTINUOUS:
                    values = [QLocale().toDouble(v)[0] for v in values]
                # None will match the value returned by all_vars.get
                encoded.append((attr, None, op, values))
            else:
                # check for exact match, pylint: disable=unidiomatic-typecheck
                if type(attr) is ContinuousVariable \
                        and values and isinstance(values[0], str):
                    values = [QLocale().toDouble(v)[0] for v in values]
                elif isinstance(attr, DiscreteVariable):
                    values = [attr.values[i - 1] if i else "" for i in values]
                encoded.append((
                    attr.name,
                    context.attributes.get(attr.name)
                    or context.metas.get(attr.name),
                    op,
                    values
                ))
        return encoded

    # pylint: disable=arguments-differ
    def decode_setting(self, setting, value, domain=None, *_args):
        value = super().decode_setting(setting, value, domain)
        if setting.name == 'conditions':
            CONTINUOUS = vartype(ContinuousVariable("x"))
            for i, (attr, tpe, op, values) in enumerate(value):
                if tpe is not None:
                    attr = domain[attr]
                # check for exact match, pylint: disable=unidiomatic-typecheck
                if type(attr) is ContinuousVariable \
                        or OWSelectRows.AllTypes.get(attr) == CONTINUOUS:
                    values = [QLocale().toString(float(i), 'f') for i in values]
                elif isinstance(attr, DiscreteVariable):
                    values = tuple(
                        attr.to_val(val) + 1 if val else 0
                        for val in values
                        if val in attr.values
                    ) or (0,)
                value[i] = (attr, op, values)
        return value

    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return self.PERFECT_MATCH

        conditions = context.values["conditions"]
        all_vars = attrs.copy()
        all_vars.update(metas)
        matched = [
            all_vars.get(name) == tpe  # also matches "all (...)" strings
            for name, tpe, *rest in conditions
        ]
        if any(matched):
            return 0.5 * sum(matched) / len(matched)
        return self.NO_MATCH

    def filter_value(self, setting, data, domain, attrs, metas):
        if setting.name != "conditions":
            super().filter_value(setting, data, domain, attrs, metas)
            return

        all_vars = attrs.copy()
        all_vars.update(metas)
        conditions = data["conditions"]
        conditions[:] = [
            (name, tpe, *rest)
            for name, tpe, *rest in conditions
            if all_vars.get(name) == tpe
        ]


class FilterDiscreteType(enum.Enum):
    # pylint: disable=invalid-name
    Equal = "Equal"
    NotEqual = "NotEqual"
    In = "In"
    IsDefined = "IsDefined"


def _plural(s):
    s = s.replace("is ", "are ")
    for word in ("equals", "contains", "begins", "ends"):
        s = s.replace(word, word[:-1])
    return s


class OWSelectRows(widget.OWWidget):
    name = "Select Rows"
    description = "Select rows from the data based on values of variables."
    icon = "icons/SelectRows.svg"
    priority = 100
    category = "Transform"
    keywords = ["filter"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        matching_data = Output("Matching Data", Table, default=True)
        unmatched_data = Output("Unmatched Data", Table)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    want_main_area = False

    settingsHandler = SelectRowsContextHandler()
    conditions = ContextSetting([])
    update_on_change = Setting(True)
    purge_attributes = Setting(False, schema_only=True)
    purge_classes = Setting(False, schema_only=True)
    auto_commit = Setting(True)

    settings_version = 2

    Operators = {
        ContinuousVariable: [
            (FilterContinuous.Equal, "equals"),
            (FilterContinuous.NotEqual, "is not"),
            (FilterContinuous.Less, "is below"),
            (FilterContinuous.LessEqual, "is at most"),
            (FilterContinuous.Greater, "is greater than"),
            (FilterContinuous.GreaterEqual, "is at least"),
            (FilterContinuous.Between, "is between"),
            (FilterContinuous.Outside, "is outside"),
            (FilterContinuous.IsDefined, "is defined"),
        ],
        DiscreteVariable: [
            (FilterDiscreteType.Equal, "is"),
            (FilterDiscreteType.NotEqual, "is not"),
            (FilterDiscreteType.In, "is one of"),
            (FilterDiscreteType.IsDefined, "is defined")
        ],
        StringVariable: [
            (FilterString.Equal, "equals"),
            (FilterString.NotEqual, "is not"),
            (FilterString.Less, "is before"),
            (FilterString.LessEqual, "is equal or before"),
            (FilterString.Greater, "is after"),
            (FilterString.GreaterEqual, "is equal or after"),
            (FilterString.Between, "is between"),
            (FilterString.Outside, "is outside"),
            (FilterString.Contains, "contains"),
            (FilterString.StartsWith, "begins with"),
            (FilterString.EndsWith, "ends with"),
            (FilterString.IsDefined, "is defined"),
        ]
    }

    Operators[TimeVariable] = Operators[ContinuousVariable]

    AllTypes = {}
    for _all_name, _all_type, _all_ops in (
            ("All variables", 0,
             [(None, "are defined")]),
            ("All numeric variables", 2,
             [(v, _plural(t)) for v, t in Operators[ContinuousVariable]]),
            ("All string variables", 3,
             [(v, _plural(t)) for v, t in Operators[StringVariable]])):
        Operators[_all_name] = _all_ops
        AllTypes[_all_name] = _all_type

    operator_names = {vtype: [name for _, name in filters]
                      for vtype, filters in Operators.items()}

    class Error(widget.OWWidget.Error):
        parsing_error = Msg("{}")

    def __init__(self):
        super().__init__()

        self.old_purge_classes = True

        self.conditions = []
        self.last_output_conditions = None
        self.data = None
        self.data_desc = self.match_desc = self.nonmatch_desc = None
        self.variable_model = DomainModel(
            [list(self.AllTypes), DomainModel.Separator,
             DomainModel.CLASSES, DomainModel.ATTRIBUTES, DomainModel.METAS])

        box = gui.vBox(self.controlArea, 'Conditions', stretch=100)
        self.cond_list = QTableWidget(
            box, showGrid=False, selectionMode=QTableWidget.NoSelection)
        box.layout().addWidget(self.cond_list)
        self.cond_list.setColumnCount(4)
        self.cond_list.setRowCount(0)
        self.cond_list.verticalHeader().hide()
        self.cond_list.horizontalHeader().hide()
        for i in range(3):
            self.cond_list.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        self.cond_list.horizontalHeader().resizeSection(3, 30)
        self.cond_list.viewport().setBackgroundRole(QPalette.Window)

        box2 = gui.hBox(box)
        gui.rubber(box2)
        self.add_button = gui.button(
            box2, self, "Add Condition", callback=self.add_row)
        self.add_all_button = gui.button(
            box2, self, "Add All Variables", callback=self.add_all)
        self.remove_all_button = gui.button(
            box2, self, "Remove All", callback=self.remove_all)
        gui.rubber(box2)

        box_setting = gui.vBox(self.buttonsArea)
        self.cb_pa = gui.checkBox(
            box_setting, self, "purge_attributes", "Remove unused features",
            callback=self.conditions_changed)
        self.cb_pc = gui.checkBox(
            box_setting, self, "purge_classes", "Remove unused classes",
            callback=self.conditions_changed)

        self.report_button.setFixedWidth(120)
        gui.rubber(self.buttonsArea.layout())

        gui.auto_send(self.buttonsArea, self, "auto_commit")

        self.set_data(None)
        self.resize(600, 400)

    def add_row(self, attr=None, condition_type=None, condition_value=None):
        model = self.cond_list.model()
        row = model.rowCount()
        model.insertRow(row)

        attr_combo = ComboBoxSearch(
            minimumContentsLength=12,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon)
        attr_combo.setModel(self.variable_model)
        attr_combo.row = row
        attr_combo.setCurrentIndex(self.variable_model.indexOf(attr) if attr
                                   else len(self.AllTypes) + 1)
        self.cond_list.setCellWidget(row, 0, attr_combo)

        index = QPersistentModelIndex(model.index(row, 3))
        temp_button = QPushButton('×', self, flat=True,
                                  styleSheet='* {font-size: 16pt; color: palette(button-text) }'
                                             '*:hover {color: palette(bright-text)}')
        temp_button.clicked.connect(lambda: self.remove_one(index.row()))
        self.cond_list.setCellWidget(row, 3, temp_button)

        self.remove_all_button.setDisabled(False)
        self.set_new_operators(attr_combo, attr is not None,
                               condition_type, condition_value)
        attr_combo.currentIndexChanged.connect(
            lambda _: self.set_new_operators(attr_combo, False))

        self.cond_list.resizeRowToContents(row)

    def add_all(self):
        if self.cond_list.rowCount():
            Mb = QMessageBox
            if Mb.question(
                    self, "Remove existing filters",
                    "This will replace the existing filters with "
                    "filters for all variables.", Mb.Ok | Mb.Cancel) != Mb.Ok:
                return
            self.remove_all()
        for attr in self.variable_model[len(self.AllTypes) + 1:]:
            self.add_row(attr)
        self.conditions_changed()

    def remove_one(self, rownum):
        self.remove_one_row(rownum)
        self.conditions_changed()

    def remove_all(self):
        self.remove_all_rows()
        self.conditions_changed()

    def remove_one_row(self, rownum):
        self.cond_list.removeRow(rownum)
        if self.cond_list.model().rowCount() == 0:
            self.remove_all_button.setDisabled(True)

    def remove_all_rows(self):
        # Disconnect signals to avoid stray emits when changing variable_model
        for row in range(self.cond_list.rowCount()):
            for col in (0, 1):
                widg = self.cond_list.cellWidget(row, col)
                if widg:
                    widg.currentIndexChanged.disconnect()
        self.cond_list.clear()
        self.cond_list.setRowCount(0)
        self.remove_all_button.setDisabled(True)

    def set_new_operators(self, attr_combo, adding_all,
                          selected_index=None, selected_values=None):
        old_combo = self.cond_list.cellWidget(attr_combo.row, 1)
        prev_text = old_combo.currentText() if old_combo else ""
        oper_combo = QComboBox()
        oper_combo.row = attr_combo.row
        oper_combo.attr_combo = attr_combo
        attr_name = attr_combo.currentText()
        if attr_name in self.AllTypes:
            oper_combo.addItems(self.operator_names[attr_name])
        else:
            var = self.data.domain[attr_name]
            oper_combo.addItems(self.operator_names[type(var)])
        if selected_index is None:
            selected_index = oper_combo.findText(prev_text)
            if selected_index == -1:
                selected_index = 0
        oper_combo.setCurrentIndex(selected_index)
        self.cond_list.setCellWidget(oper_combo.row, 1, oper_combo)
        self.set_new_values(oper_combo, adding_all, selected_values)
        oper_combo.currentIndexChanged.connect(
            lambda _: self.set_new_values(oper_combo, False))

    @staticmethod
    def _get_lineedit_contents(box):
        contents = []
        for child in getattr(box, "controls", [box]):
            if isinstance(child, QLineEdit):
                contents.append(child.text())
            elif isinstance(child, DateTimeWidget):
                if child.format == (0, 1):
                    contents.append(child.time())
                elif child.format == (1, 0):
                    contents.append(child.date())
                elif child.format == (1, 1):
                    contents.append(child.dateTime())
        return contents

    @staticmethod
    def _get_value_contents(box):
        cont = []
        names = []
        for child in getattr(box, "controls", [box]):
            if isinstance(child, QLineEdit):
                cont.append(child.text())
            elif isinstance(child, QComboBox):
                cont.append(child.currentIndex())
            elif isinstance(child, QToolButton):
                if child.popup is not None:
                    model = child.popup.list_view.model()
                    for row in range(model.rowCount()):
                        item = model.item(row)
                        if item.checkState():
                            cont.append(row + 1)
                            names.append(item.text())
                    child.desc_text = ', '.join(names)
                    child.set_text()
            elif isinstance(child, DateTimeWidget):
                if child.format == (0, 1):
                    cont.append(child.time())
                elif child.format == (1, 0):
                    cont.append(child.date())
                elif child.format == (1, 1):
                    cont.append(child.dateTime())
            elif isinstance(child, QLabel) or child is None:
                pass
            else:
                raise TypeError('Type %s not supported.' % type(child))
        return tuple(cont)

    class QDoubleValidatorEmpty(QDoubleValidator):
        def validate(self, input_, pos):
            if not input_:
                return QDoubleValidator.Acceptable, input_, pos
            if self.locale().groupSeparator() in input_:
                return QDoubleValidator.Invalid, input_, pos
            return super().validate(input_, pos)

    def set_new_values(self, oper_combo, adding_all, selected_values=None):
        # def remove_children():
        #     for child in box.children()[1:]:
        #         box.layout().removeWidget(child)
        #         child.setParent(None)

        def add_textual(contents):
            le = gui.lineEdit(box, self, None,
                              sizePolicy=QSizePolicy(QSizePolicy.Expanding,
                                                     QSizePolicy.Expanding))
            if contents:
                le.setText(contents)
            le.setAlignment(Qt.AlignRight)
            le.editingFinished.connect(self.conditions_changed)
            return le

        def add_numeric(contents):
            le = add_textual(contents)
            le.setValidator(OWSelectRows.QDoubleValidatorEmpty())
            return le

        box = self.cond_list.cellWidget(oper_combo.row, 2)
        lc = ["", ""]
        oper = oper_combo.currentIndex()
        attr_name = oper_combo.attr_combo.currentText()
        if attr_name in self.AllTypes:
            vtype = self.AllTypes[attr_name]
            var = None
        else:
            var = self.data.domain[attr_name]
            var_idx = self.data.domain.index(attr_name)
            vtype = vartype(var)
            if selected_values is not None:
                lc = list(selected_values) + ["", ""]
                lc = [str(x) if vtype != 4 else x for x in lc[:2]]
        if box and vtype == box.var_type:
            lc = self._get_lineedit_contents(box) + lc

        if oper_combo.currentText().endswith(" defined"):
            label = QLabel()
            label.var_type = vtype
            self.cond_list.setCellWidget(oper_combo.row, 2, label)
        elif var is not None and var.is_discrete:
            if oper_combo.currentText().endswith(" one of"):
                if selected_values:
                    lc = list(selected_values)
                button = DropDownToolButton(self, var, lc)
                button.var_type = vtype
                self.cond_list.setCellWidget(oper_combo.row, 2, button)
            else:
                combo = ComboBoxSearch()
                combo.addItems(("", ) + var.values)
                if lc[0]:
                    combo.setCurrentIndex(int(lc[0]))
                else:
                    combo.setCurrentIndex(0)
                combo.var_type = vartype(var)
                self.cond_list.setCellWidget(oper_combo.row, 2, combo)
                combo.currentIndexChanged.connect(self.conditions_changed)
        else:
            box = gui.hBox(self, addToLayout=False)
            box.var_type = vtype
            self.cond_list.setCellWidget(oper_combo.row, 2, box)
            if vtype == 2:  # continuous:
                box.controls = [add_numeric(lc[0])]
                if oper > 5:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(add_numeric(lc[1]))
            elif vtype == 3:  # string:
                box.controls = [add_textual(lc[0])]
                if oper in [6, 7]:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(add_textual(lc[1]))
            elif vtype == 4:  # time:
                def invalidate_datetime():
                    if w_:
                        if w.dateTime() > w_.dateTime():
                            w_.setDateTime(w.dateTime())
                        if w.format == (1, 1):
                            w.calendarWidget.timeedit.setTime(w.time())
                            w_.calendarWidget.timeedit.setTime(w_.time())
                    elif w.format == (1, 1):
                        w.calendarWidget.timeedit.setTime(w.time())

                def datetime_changed():
                    self.conditions_changed()
                    invalidate_datetime()

                datetime_format = (var.have_date, var.have_time)
                column = self.data.get_column_view(var_idx)[0]
                w = DateTimeWidget(self, column, datetime_format)
                w.set_datetime(lc[0])
                box.controls = [w]
                box.layout().addWidget(w)
                w.dateTimeChanged.connect(datetime_changed)
                if oper > 5:
                    gui.widgetLabel(box, " and ")
                    w_ = DateTimeWidget(self, column, datetime_format)
                    w_.set_datetime(lc[1])
                    box.layout().addWidget(w_)
                    box.controls.append(w_)
                    invalidate_datetime()
                    w_.dateTimeChanged.connect(datetime_changed)
                else:
                    w_ = None
            else:
                box.controls = []
        if not adding_all:
            self.conditions_changed()

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data
        self.cb_pa.setEnabled(not isinstance(data, SqlTable))
        self.cb_pc.setEnabled(not isinstance(data, SqlTable))
        self.remove_all_rows()
        self.add_button.setDisabled(data is None)
        self.add_all_button.setDisabled(
            data is None or
            len(data.domain.variables) + len(data.domain.metas) > 100)
        if not data:
            self.data_desc = None
            self.variable_model.set_domain(None)
            self.commit.deferred()
            return
        self.data_desc = report.describe_data_brief(data)
        self.variable_model.set_domain(data.domain)

        self.conditions = []
        self.openContext(data)
        for attr, cond_type, cond_value in self.conditions:
            if attr in self.variable_model:
                self.add_row(attr, cond_type, cond_value)
        if not self.cond_list.model().rowCount():
            self.add_row()

        self.commit.now()

    def conditions_changed(self):
        try:
            cells_by_rows = (
                [self.cond_list.cellWidget(row, col) for col in range(3)]
                for row in range(self.cond_list.rowCount())
            )
            self.conditions = [
                (var_cell.currentData(gui.TableVariable) or var_cell.currentText(),
                 oper_cell.currentIndex(),
                 self._get_value_contents(val_cell))
                for var_cell, oper_cell, val_cell in cells_by_rows]
            if self.update_on_change and (
                    self.last_output_conditions is None or
                    self.last_output_conditions != self.conditions):
                self.commit.deferred()
        except AttributeError:
            # Attribute error appears if the signal is triggered when the
            # controls are being constructed
            pass

    @staticmethod
    def _values_to_floats(attr, values):
        if len(values) == 0:
            return values
        if not all(values):
            return None
        if isinstance(attr, TimeVariable):
            values = (value.toString(format=Qt.ISODate) for value in values)
            parse = lambda x: (attr.parse(x), True)
        else:
            parse = QLocale().toDouble

        try:
            floats, ok = zip(*[parse(v) for v in values])
            if not all(ok):
                raise ValueError('Some values could not be parsed as floats'
                                 'in the current locale: {}'.format(values))
        except TypeError:
            floats = values  # values already floats
        assert all(isinstance(v, float) for v in floats)
        return floats

    @gui.deferred
    def commit(self):
        matching_output = self.data
        non_matching_output = None
        annotated_output = None

        self.Error.clear()
        if self.data:
            domain = self.data.domain
            conditions = []
            for attr_name, oper_idx, values in self.conditions:
                if attr_name in self.AllTypes:
                    attr_index = attr = None
                    attr_type = self.AllTypes[attr_name]
                    operators = self.Operators[attr_name]
                else:
                    attr_index = domain.index(attr_name)
                    attr = domain[attr_index]
                    attr_type = vartype(attr)
                    operators = self.Operators[type(attr)]
                opertype, _ = operators[oper_idx]
                if attr_type == 0:
                    filt = data_filter.IsDefined()
                elif attr_type in (2, 4):  # continuous, time
                    try:
                        floats = self._values_to_floats(attr, values)
                    except ValueError as e:
                        self.Error.parsing_error(e.args[0])
                        return
                    if floats is None:
                        continue
                    filt = data_filter.FilterContinuous(
                        attr_index, opertype, *floats)
                elif attr_type == 3:  # string
                    filt = data_filter.FilterString(
                        attr_index, opertype, *[str(v) for v in values])
                else:
                    if opertype == FilterDiscreteType.IsDefined:
                        f_values = None
                    else:
                        if not values or not values[0]:
                            continue
                        values = [attr.values[i-1] for i in values]
                        if opertype == FilterDiscreteType.Equal:
                            f_values = {values[0]}
                        elif opertype == FilterDiscreteType.NotEqual:
                            f_values = set(attr.values)
                            f_values.remove(values[0])
                        elif opertype == FilterDiscreteType.In:
                            f_values = set(values)
                        else:
                            raise ValueError("invalid operand")
                    filt = data_filter.FilterDiscrete(attr_index, f_values)
                conditions.append(filt)

            if conditions:
                filters = data_filter.Values(conditions)
                matching_output = filters(self.data)
                filters.negate = True
                non_matching_output = filters(self.data)

                row_sel = np.in1d(self.data.ids, matching_output.ids)
                annotated_output = create_annotated_table(self.data, row_sel)

            # if hasattr(self.data, "name"):
            #     matching_output.name = self.data.name
            #     non_matching_output.name = self.data.name

            purge_attrs = self.purge_attributes
            purge_classes = self.purge_classes
            if (purge_attrs or purge_classes) and \
                    not isinstance(self.data, SqlTable):
                attr_flags = sum([Remove.RemoveConstant * purge_attrs,
                                  Remove.RemoveUnusedValues * purge_attrs])
                class_flags = sum([Remove.RemoveConstant * purge_classes,
                                   Remove.RemoveUnusedValues * purge_classes])
                # same settings used for attributes and meta features
                remover = Remove(attr_flags, class_flags, attr_flags)

                matching_output = remover(matching_output)
                non_matching_output = remover(non_matching_output)
                annotated_output = remover(annotated_output)

        if not matching_output:
            matching_output = None
        if not non_matching_output:
            non_matching_output = None
        if not annotated_output:
            annotated_output = None

        self.Outputs.matching_data.send(matching_output)
        self.Outputs.unmatched_data.send(non_matching_output)
        self.Outputs.annotated_data.send(annotated_output)

        self.match_desc = report.describe_data_brief(matching_output)
        self.nonmatch_desc = report.describe_data_brief(non_matching_output)

    def send_report(self):
        if not self.data:
            self.report_paragraph("No data.")
            return

        pdesc = None
        describe_domain = False
        for d in (self.data_desc, self.match_desc, self.nonmatch_desc):
            if not d or not d["Data instances"]:
                continue
            ndesc = d.copy()
            del ndesc["Data instances"]
            if pdesc is not None and pdesc != ndesc:
                describe_domain = True
            pdesc = ndesc

        conditions = []
        for attr, oper, values in self.conditions:
            if isinstance(attr, str):
                attr_name = attr
                var_type = self.AllTypes[attr]
                names = self.operator_names[attr_name]
            else:
                attr_name = attr.name
                var_type = vartype(attr)
                names = self.operator_names[type(attr)]
            name = names[oper]
            if oper == len(names) - 1:
                conditions.append("{} {}".format(attr_name, name))
            elif var_type == 1:  # discrete
                if name == "is one of":
                    valnames = [attr.values[v - 1] for v in values]
                    if not valnames:
                        continue
                    if len(valnames) == 1:
                        valstr = valnames[0]
                    else:
                        valstr = f"{', '.join(valnames[:-1])} or {valnames[-1]}"
                    conditions.append(f"{attr} is {valstr}")
                elif values and values[0]:
                    value = values[0] - 1
                    conditions.append(f"{attr} {name} {attr.values[value]}")
            elif var_type == 3:  # string variable
                conditions.append(
                    f"{attr} {name} {' and '.join(map(repr, values))}")
            elif var_type == 4:  # time
                values = (value.toString(format=Qt.ISODate) for value in values)
                conditions.append(f"{attr} {name} {' and '.join(values)}")
            elif all(x for x in values):  # numeric variable
                conditions.append(f"{attr} {name} {' and '.join(values)}")
        items = OrderedDict()
        if describe_domain:
            items.update(self.data_desc)
        else:
            items["Instances"] = self.data_desc["Data instances"]
        items["Condition"] = " AND ".join(conditions) or "no conditions"
        self.report_items("Data", items)
        if describe_domain:
            self.report_items("Matching data", self.match_desc)
            self.report_items("Non-matching data", self.nonmatch_desc)
        else:
            match_inst = \
                bool(self.match_desc) and \
                self.match_desc["Data instances"]
            nonmatch_inst = \
                bool(self.nonmatch_desc) and \
                self.nonmatch_desc["Data instances"]
            self.report_items(
                "Output",
                (("Matching data",
                  "{} instances".format(match_inst) if match_inst else "None"),
                 ("Non-matching data",
                  nonmatch_inst > 0 and "{} instances".format(nonmatch_inst))))

    @classmethod
    def migrate_context(cls, context, version):
        if not version or version < 2:
            # Just remove; can't migrate because variables types are unknown
            context.values["conditions"] = []


class CheckBoxPopup(QWidget):
    def __init__(self, var, lc, widget_parent=None, widg=None):
        QWidget.__init__(self)

        self.list_view = QListView()
        text = []
        model = QStandardItemModel(self.list_view)
        for (i, val) in enumerate(var.values):
            item = QStandardItem(val)
            item.setCheckable(True)
            if i + 1 in lc:
                item.setCheckState(Qt.Checked)
                text.append(val)
            model.appendRow(item)
        model.itemChanged.connect(widget_parent.conditions_changed)
        self.list_view.setModel(model)

        layout = QGridLayout(self)
        layout.addWidget(self.list_view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.adjustSize()
        self.setWindowFlags(Qt.Popup)

        self.widget = widg
        self.widget.desc_text = ', '.join(text)
        self.widget.set_text()

    def moved(self):
        point = self.widget.rect().bottomRight()
        global_point = self.widget.mapToGlobal(point)
        self.move(global_point - QPoint(self.width(), 0))


class DropDownToolButton(QToolButton):
    def __init__(self, parent, var, lc):
        QToolButton.__init__(self, parent)
        self.desc_text = ''
        self.popup = CheckBoxPopup(var, lc, parent, self)
        self.setMenu(QMenu()) # to show arrow
        self.clicked.connect(self.open_popup)

    def open_popup(self):
        self.popup.moved()
        self.popup.show()

    def set_text(self):
        metrics = QFontMetrics(self.font())
        self.setText(metrics.elidedText(self.desc_text, Qt.ElideRight,
                                        self.width() - 15))

    def resizeEvent(self, _):
        self.set_text()


class DateTimeWidget(QDateTimeEdit):
    def __init__(self, parent, column, datetime_format):
        QDateTimeEdit.__init__(self, parent)

        self.format = datetime_format
        self.have_date, self.have_time = datetime_format[0], datetime_format[1]
        self.set_format(column)
        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

    def set_format(self, column):
        str_format = Qt.ISODate
        if self.have_date and self.have_time:
            self.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
            c_format = "%Y-%m-%d %H:%M:%S"
            min_datetime, max_datetime = self.find_range(column, c_format)
            self.min_datetime = QDateTime.fromString(min_datetime, str_format)
            self.max_datetime = QDateTime.fromString(max_datetime, str_format)
            self.setCalendarPopup(True)
            self.calendarWidget = gui.CalendarWidgetWithTime(
                self, time=self.min_datetime.time())
            self.calendarWidget.timeedit.timeChanged.connect(
                self.set_datetime)
            self.setCalendarWidget(self.calendarWidget)
            self.setDateTimeRange(self.min_datetime, self.max_datetime)

        elif self.have_date and not self.have_time:
            self.setDisplayFormat("yyyy-MM-dd")
            self.setCalendarPopup(True)
            min_datetime, max_datetime = self.find_range(column, "%Y-%m-%d")
            self.min_datetime = QDate.fromString(min_datetime, str_format)
            self.max_datetime = QDate.fromString(max_datetime, str_format)
            self.setDateRange(self.min_datetime, self.max_datetime)

        elif not self.have_date and self.have_time:
            self.setDisplayFormat("hh:mm:ss")
            min_datetime, max_datetime = self.find_range(column, "%H:%M:%S")
            self.min_datetime = QTime.fromString(min_datetime, str_format)
            self.max_datetime = QTime.fromString(max_datetime, str_format)
            self.setTimeRange(self.min_datetime, self.max_datetime)

    def set_datetime(self, date_time):
        if not date_time:
            date_time = self.min_datetime
        if self.have_date and self.have_time:
            if isinstance(date_time, QTime):
                self.setDateTime(
                    QDateTime(self.date(), self.calendarWidget.timeedit.time()))
            else:
                self.setDateTime(date_time)
        elif self.have_date and not self.have_time:
            self.setDate(date_time)
        elif not self.have_date and self.have_time:
            self.setTime(date_time)

    @staticmethod
    def find_range(column, convert_format):
        def convert_timestamp(timestamp):
            if timestamp >= 0:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return datetime(1970, 1, 1, tzinfo=timezone.utc) + \
                       timedelta(seconds=int(timestamp))

        min_datetime = convert_timestamp(
            np.nanmin(column)).strftime(convert_format)
        max_datetime = convert_timestamp(
            np.nanmax(column)).strftime(convert_format)
        return min_datetime, max_datetime


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSelectRows).run(Table("heart_disease"))
