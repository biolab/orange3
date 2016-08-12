from collections import OrderedDict
from itertools import chain

import operator
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import numpy as np

from Orange.data import (ContinuousVariable, DiscreteVariable, StringVariable,
                         Table, TimeVariable, TableBase)
from Orange.data.domain import filter_visible
from Orange.data.sql.table import SqlTable
from Orange.preprocess import Remove
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils import vartype
from Orange.canvas import report


class SelectRowsContextHandler(DomainContextHandler):
    """Context handler that filters conditions"""

    def is_valid_item(self, setting, condition, attrs, metas):
        """Return True if condition applies to a variable in given domain."""
        varname, *_ = condition
        return varname in attrs or varname in metas


class Filter:
    """
    Provides a list of available filters and a mapping to their functionality and
    the variables which it can process.
    filter_function must accept arguments in the same order as they are given in
    the GUI, with the column (i.e. a pandas Series) as the first argument.
    """
    values = []
    _variable_bindings = {}

    def __init__(self, filter_text, filter_function):
        self.filter_text = filter_text
        self.filter_function = filter_function
        self.supported_variables = []

        # register to the static values collection here, cleaner and safer
        Filter.values.append(self)

    def __call__(self, *args, **kwargs):
        return self.filter_function(*args, **kwargs)

    def __str__(self):
        return self.filter_text

    @classmethod
    def for_variable(cls, var):
        return cls._variable_bindings[var]

    @classmethod
    def bind(cls, var, filters):
        """Bind a list of filters to a variable type. """
        cls._variable_bindings[var] = filters
        for f in filters:
            f.supported_variables.append(var)

Filter.Equals = Filter("equals", operator.eq)
Filter.IsNot = Filter("is not", operator.ne)
Filter.IsBelow = Filter("is below", operator.lt)
Filter.IsAtMost = Filter("is at most", operator.le)
Filter.IsGreaterThan = Filter("is greater than", operator.gt)
Filter.IsAtLeast = Filter("is at least", operator.ge)
Filter.IsBetween = Filter("is between", lambda col, low, high: (low <= col) & (col <= high))
Filter.IsOutside = Filter("is outside", lambda col, low, high: ~((low <= col) & (col <= high)))
Filter.IsDefined = Filter("is defined", lambda col: ~np.isnan(col))
Filter.Is = Filter("is", operator.eq)
Filter.IsOneOf = Filter("is one of", lambda col, vals: col.apply(lambda el: el in vals))
Filter.IsBefore = Filter("is before", operator.lt)
Filter.IsEqualOrBefore = Filter("is equal or before", operator.le)
Filter.IsAfter = Filter("is after", operator.gt)
Filter.IsEqualOrAfter = Filter("is equal or after", operator.ge)
Filter.Contains = Filter("contains", lambda col, what: col.apply(lambda el: what in el))
Filter.BeginsWith = Filter("begins with", lambda col, what: col.apply(lambda el: el.startswith(what)))
Filter.EndsWith = Filter("ends with", lambda col, what: col.apply(lambda el: el.endswith(what)))

# bindings here for code clarity and to allow completely independent ordering for the GUI
Filter.bind(ContinuousVariable, [Filter.Equals, Filter.IsNot, Filter.IsBelow, Filter.IsAtMost,
                                 Filter.IsGreaterThan, Filter.IsAtLeast, Filter.IsBetween,
                                 Filter.IsOutside, Filter.IsDefined])
Filter.bind(DiscreteVariable, [Filter.Is, Filter.IsNot, Filter.IsOneOf, Filter.IsDefined])
Filter.bind(StringVariable, [Filter.Equals, Filter.IsNot, Filter.IsBefore, Filter.IsEqualOrBefore,
                             Filter.IsAfter, Filter.IsEqualOrAfter, Filter.IsBetween, Filter.IsOutside,
                             Filter.Contains, Filter.BeginsWith, Filter.EndsWith, Filter.IsDefined])
Filter.bind(TimeVariable, Filter.for_variable(ContinuousVariable))


class OWSelectRows(widget.OWWidget):
    name = "Select Rows"
    id = "Orange.widgets.data.file"
    description = "Select rows from the data based on values of variables."
    icon = "icons/SelectRows.svg"
    priority = 100
    category = "Data"
    inputs = [("Data", TableBase, "set_data")]
    outputs = [("Matching Data", TableBase, widget.Default), ("Unmatched Data", TableBase)]

    want_main_area = False

    settingsHandler = SelectRowsContextHandler()
    conditions = ContextSetting([])
    update_on_change = Setting(True)
    purge_attributes = Setting(True)
    purge_classes = Setting(True)
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()

        self.old_purge_classes = True

        self.conditions = []
        self.last_output_conditions = None
        self.data = None
        self.data_desc = self.match_desc = self.nonmatch_desc = None

        box = gui.vBox(self.controlArea, 'Conditions', stretch=100)
        self.cond_list = QtGui.QTableWidget(
            box, showGrid=False, selectionMode=QtGui.QTableWidget.NoSelection)
        box.layout().addWidget(self.cond_list)
        self.cond_list.setColumnCount(3)
        self.cond_list.setRowCount(0)
        self.cond_list.verticalHeader().hide()
        self.cond_list.horizontalHeader().hide()
        self.cond_list.resizeColumnToContents(0)
        self.cond_list.horizontalHeader().setResizeMode(
            QtGui.QHeaderView.Stretch)
        self.cond_list.viewport().setBackgroundRole(QtGui.QPalette.Window)

        box2 = gui.hBox(box)
        gui.rubber(box2)
        self.add_button = gui.button(
            box2, self, "Add Condition", callback=self.add_row)
        self.add_all_button = gui.button(
            box2, self, "Add All Variables", callback=self.add_all)
        self.remove_all_button = gui.button(
            box2, self, "Remove All", callback=self.remove_all)
        gui.rubber(box2)

        boxes = gui.widgetBox(self.controlArea, orientation=QtGui.QGridLayout())
        layout = boxes.layout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        box_data = gui.vBox(boxes, 'Data', addToLayout=False)
        self.data_in_variables = gui.widgetLabel(box_data, " ")
        self.data_out_rows = gui.widgetLabel(box_data, " ")
        layout.addWidget(box_data, 0, 0)

        box_setting = gui.vBox(boxes, 'Purging', addToLayout=False)
        self.cb_pa = gui.checkBox(
            box_setting, self, "purge_attributes", "Remove unused features",
            callback=self.conditions_changed)
        gui.separator(box_setting, height=1)
        self.cb_pc = gui.checkBox(
            box_setting, self, "purge_classes", "Remove unused classes",
            callback=self.conditions_changed)
        layout.addWidget(box_setting, 0, 1)

        self.report_button.setFixedWidth(120)
        gui.rubber(self.buttonsArea.layout())
        layout.addWidget(self.buttonsArea, 1, 0)

        acbox = gui.auto_commit(
            None, self, "auto_commit", label="Send", orientation=Qt.Horizontal,
            checkbox_label="Send automatically")
        layout.addWidget(acbox, 1, 1)

        self.set_data(None)
        self.resize(600, 400)

    def add_row(self, attr=None, condition_type=None, condition_value=None):
        model = self.cond_list.model()
        row = model.rowCount()
        model.insertRow(row)

        attr_combo = QtGui.QComboBox(
            minimumContentsLength=12,
            sizeAdjustPolicy=QtGui.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        attr_combo.row = row
        for var in filter_visible(chain(self.data.domain.variables, self.data.domain.metas)):
            attr_combo.addItem(*gui.attributeItem(var))
        attr_combo.setCurrentIndex(attr or 0)
        self.cond_list.setCellWidget(row, 0, attr_combo)

        self.remove_all_button.setDisabled(False)
        self.set_new_operators(attr_combo, attr is not None,
                               condition_type, condition_value)
        attr_combo.currentIndexChanged.connect(
            lambda _: self.set_new_operators(attr_combo, False))

        self.cond_list.resizeRowToContents(row)

    def add_all(self):
        if self.cond_list.rowCount():
            Mb = QtGui.QMessageBox
            if Mb.question(
                    self, "Remove existing filters",
                    "This will replace the existing filters with "
                    "filters for all variables.", Mb.Ok | Mb.Cancel) != Mb.Ok:
                return
            self.remove_all()
        domain = self.data.domain
        for i in range(len(domain.variables) + len(domain.metas)):
            self.add_row(i)

    def remove_all(self):
        self.remove_all_rows()
        self.conditions_changed()

    def remove_all_rows(self):
        self.cond_list.clear()
        self.cond_list.setRowCount(0)
        self.remove_all_button.setDisabled(True)

    def set_new_operators(self, attr_combo, adding_all,
                          selected_index=None, selected_values=None):
        oper_combo = QtGui.QComboBox()
        oper_combo.row = attr_combo.row
        oper_combo.attr_combo = attr_combo
        var = self.data.domain[attr_combo.currentText()]
        oper_combo.addItems([str(f) for f in Filter.for_variable(type(var))])
        oper_combo.setCurrentIndex(selected_index or 0)
        self.set_new_values(oper_combo, adding_all, selected_values)
        self.cond_list.setCellWidget(oper_combo.row, 1, oper_combo)
        oper_combo.currentIndexChanged.connect(
            lambda _: self.set_new_values(oper_combo, False))

    @staticmethod
    def _get_lineedit_contents(box):
        return [child.text() for child in getattr(box, "controls", [box])
                if isinstance(child, QtGui.QLineEdit)]

    def _get_value_contents(self, box):
        cont = []
        names = []
        for child in getattr(box, "controls", [box]):
            if isinstance(child, QtGui.QLineEdit):
                cont.append(child.text())
            elif isinstance(child, QtGui.QComboBox):
                # use the actual discrete variable value, not the .values index
                cont.append(child.currentText())
            elif isinstance(child, QtGui.QToolButton):
                if child.popup is not None:
                    model = child.popup.list_view.model()
                    for row in range(model.rowCount()):
                        item = model.item(row)
                        if item.checkState():
                            names.append(item.text())
                    child.desc_text = ', '.join(names)
                    child.set_text()
                cont.append(names)
            elif child is None:
                pass
            else:
                raise TypeError('Type %s not supported.' % type(child))
        return tuple(cont)

    class QDoubleValidatorEmpty(QtGui.QDoubleValidator):
        def validate(self, input_, pos):
            if not input_:
                return (QtGui.QDoubleValidator.Acceptable, input_, pos)
            else:
                return super().validate(input_, pos)

    def set_new_values(self, oper_combo, adding_all, selected_values=None):
        # def remove_children():
        #     for child in box.children()[1:]:
        #         box.layout().removeWidget(child)
        #         child.setParent(None)

        def add_textual(contents):
            le = gui.lineEdit(box, self, None)
            if contents:
                le.setText(contents)
            le.setAlignment(QtCore.Qt.AlignRight)
            le.editingFinished.connect(self.conditions_changed)
            return le

        def add_numeric(contents):
            le = add_textual(contents)
            le.setValidator(OWSelectRows.QDoubleValidatorEmpty())
            return le

        def add_datetime(contents):
            le = add_textual(contents)
            le.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp(TimeVariable.REGEX)))
            return le

        var = self.data.domain[oper_combo.attr_combo.currentText()]
        box = self.cond_list.cellWidget(oper_combo.row, 2)
        if selected_values is not None:
            lc = list(selected_values) + ["", ""]
            lc = [str(x) for x in lc[:2]]
        else:
            lc = ["", ""]
        if box and vartype(var) == box.var_type:
            lc = self._get_lineedit_contents(box) + lc
        oper = oper_combo.currentIndex()

        if oper == oper_combo.count() - 1:
            self.cond_list.removeCellWidget(oper_combo.row, 2)
        elif var.is_discrete:
            if oper_combo.currentText() == "is one of":
                if selected_values:
                    lc = [x for x in list(selected_values)]
                button = DropDownToolButton(self, var, lc)
                button.var_type = vartype(var)
                self.cond_list.setCellWidget(oper_combo.row, 2, button)
            else:
                combo = QtGui.QComboBox()
                combo.addItems([""] + var.values)
                if lc[0]:
                    combo.setCurrentIndex(int(var.to_val(lc[0])))
                else:
                    combo.setCurrentIndex(0)
                combo.var_type = vartype(var)
                self.cond_list.setCellWidget(oper_combo.row, 2, combo)
                combo.currentIndexChanged.connect(self.conditions_changed)
        else:
            box = gui.hBox(self, addToLayout=False)
            box.var_type = vartype(var)
            self.cond_list.setCellWidget(oper_combo.row, 2, box)
            if var.is_continuous:
                validator = add_datetime if isinstance(var, TimeVariable) else add_numeric
                box.controls = [validator(lc[0])]
                if oper > 5:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(validator(lc[1]))
                gui.rubber(box)
            elif var.is_string:
                box.controls = [add_textual(lc[0])]
                if oper in [6, 7]:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(add_textual(lc[1]))
            else:
                box.controls = []
        if not adding_all:
            self.conditions_changed()

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
        if data is None:
            self.data_desc = None
            self.commit()
            return
        self.data_desc = report.describe_data_brief(data)
        self.conditions = []
        try:
            self.openContext(data)
        except Exception:
            pass

        if not self.conditions and len(data.domain.variables):
            self.add_row()
        self.update_info(data, self.data_in_variables, "In: ")
        for attr, cond_type, cond_value in self.conditions:
            attrs = [a.name for a in
                     filter_visible(chain(data.domain.variables, data.domain.metas))]
            if attr in attrs:
                self.add_row(attrs.index(attr), cond_type, cond_value)
        self.unconditional_commit()

    def conditions_changed(self):
        try:
            self.conditions = []
            self.conditions = [
                (self.cond_list.cellWidget(row, 0).currentText(),  # column name
                 self.cond_list.cellWidget(row, 1).currentIndex(),  # dropdown index
                 self._get_value_contents(self.cond_list.cellWidget(row, 2)))  # arguments
                for row in range(self.cond_list.rowCount())]
            # transform values of continuous variables into floats,
            # but not time variables (those can be compared as text)
            for i in range(len(self.conditions)):
                var = self.data.domain[self.conditions[i][0]]
                if var.is_continuous and not isinstance(var, TimeVariable):
                    self.conditions[i] = (
                        self.conditions[i][0],
                        self.conditions[i][1],
                        tuple(float(v) if v != '' else np.nan for v in self.conditions[i][2])
                    )
            if self.update_on_change and (
                    self.last_output_conditions is None or
                    self.last_output_conditions != self.conditions):
                self.commit()
        except AttributeError:
            # Attribute error appears if the signal is triggered when the
            # controls are being constructed
            pass

    def commit(self):
        self.error()
        matching_output = self.data
        non_matching_output = None
        if self.data is not None:
            # bool element-wise filter (for subscripting)
            subscript_filter = np.repeat(True, len(self.data))

            # operator_index is the index of the operation in Filter.for_variable(...)
            # because they are inserted into the dropdown in the same order
            for column, operator_index, filter_args in self.conditions:
                filter_op = Filter.for_variable(type(self.data.domain[column]))[operator_index]

                # add (element-wise and) the filter constraints to the current filter
                subscript_filter &= filter_op(self.data[column], *filter_args)

            matching_output = self.data[subscript_filter]
            non_matching_output = self.data[~subscript_filter]

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

        self.send("Matching Data", matching_output)
        self.send("Unmatched Data", non_matching_output)

        self.match_desc = report.describe_data_brief(matching_output)
        self.nonmatch_desc = report.describe_data_brief(non_matching_output)

        self.update_info(matching_output, self.data_out_rows, "Out: ")

    def update_info(self, data, lab1, label):
        def sp(s, capitalize=True):
            return s and s or ("No" if capitalize else "no"), "s" * (s != 1)

        if data is None:
            lab1.setText("")
        else:
            lab1.setText(label + "~%s row%s, %s variable%s" %
                         (sp(data.approx_len()) +
            sp(len(data.domain.variables) + len(data.domain.metas))))

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
        for column, operator_index, filter_args in self.conditions:
            filters = Filter.for_variable(type(self.data.domain[column]))
            filter_op = filters[operator_index]
            if filter_op == Filter.IsDefined:
                conditions.append("{} {}".format(column, filter_op))
            elif column.is_discrete:
                if filter_op == Filter.IsOneOf:
                    if len(filter_args) == 1:
                        conditions.append("{} is {}".format(
                            column, column.values[filter_args[0] - 1]))
                    elif len(filter_args) > 1:
                        conditions.append("{} is {} or {}".format(
                            column,
                            ", ".join(column.values[v - 1] for v in filter_args[:-1]),
                            column.values[filter_args[-1] - 1]))
                else:  # not Filter.IsOneOf
                    if not (filter_args and filter_args[0]):
                        continue
                    value = filter_args[0] - 1
                    conditions.append("{} {} {}".
                                      format(column, filter_op, column.values[value]))
            else:
                if len(filter_args) == 1:
                    conditions.append("{} {} {}".
                                      format(column, filter_op, *filter_args))
                else:
                    conditions.append("{} {} {} and {}".
                                      format(column, filter_op, *filter_args))
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


class CheckBoxPopup(QtGui.QWidget):
    def __init__(self, var, lc, widget_parent=None, widget=None):
        QtGui.QWidget.__init__(self)

        self.list_view = QtGui.QListView()
        text = []
        model = QtGui.QStandardItemModel(self.list_view)
        for (i, val) in enumerate(var.values):
            item = QtGui.QStandardItem(val)
            item.setCheckable(True)
            if i + 1 in lc:
                item.setCheckState(QtCore.Qt.Checked)
                text.append(val)
            model.appendRow(item)
        model.itemChanged.connect(widget_parent.conditions_changed)
        self.list_view.setModel(model)

        layout = QtGui.QGridLayout(self)
        layout.addWidget(self.list_view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.adjustSize()
        self.setWindowFlags(QtCore.Qt.Popup)

        self.widget = widget
        self.widget.desc_text = ', '.join(text)
        self.widget.set_text()

    def moved(self):
        point = self.widget.rect().bottomRight()
        global_point = self.widget.mapToGlobal(point)
        self.move(global_point - QtCore.QPoint(self.width(), 0))


class DropDownToolButton(QtGui.QToolButton):
    def __init__(self, parent, var, lc):
        QtGui.QToolButton.__init__(self, parent)
        self.desc_text = ''
        self.popup = CheckBoxPopup(var, lc, parent, self)
        self.setMenu(QtGui.QMenu()) # to show arrow
        self.clicked.connect(self.open_popup)

    def open_popup(self):
        self.popup.moved()
        self.popup.show()

    def set_text(self):
        metrics = QtGui.QFontMetrics(self.font())
        self.setText(metrics.elidedText(self.desc_text,
                                        QtCore.Qt.ElideRight,
                                        self.width() - 15))

    def resizeEvent(self, QResizeEvent):
        self.set_text()


def test():
    app = QtGui.QApplication([])
    w = OWSelectRows()
    w.set_data(Table("zoo"))
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
