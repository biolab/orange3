import sys
import enum
from collections import OrderedDict
from itertools import chain

import numpy as np

from AnyQt.QtWidgets import (
    QWidget, QTableWidget, QHeaderView, QComboBox, QLineEdit, QToolButton,
    QMessageBox, QMenu, QListView, QGridLayout, QPushButton, QSizePolicy,
    QLabel)
from AnyQt.QtGui import (
    QDoubleValidator, QRegExpValidator, QStandardItemModel, QStandardItem,
    QFontMetrics, QPalette
)
from AnyQt.QtCore import Qt, QPoint, QRegExp, QPersistentModelIndex, QLocale

from Orange.data import (ContinuousVariable, DiscreteVariable, StringVariable,
                         Table, TimeVariable)
import Orange.data.filter as data_filter
from Orange.data.filter import FilterContinuous, FilterString
from Orange.data.domain import filter_visible
from Orange.data.sql.table import SqlTable
from Orange.preprocess import Remove
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils import vartype
from Orange.widgets import report
from Orange.widgets.widget import Msg
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)


class SelectRowsContextHandler(DomainContextHandler):
    """Context handler that filters conditions"""

    def is_valid_item(self, setting, condition, attrs, metas):
        """Return True if condition applies to a variable in given domain."""
        varname, *_ = condition
        return varname in attrs or varname in metas

    def encode_setting(self, context, setting, value):
        if setting.name == 'conditions':
            CONTINUOUS = vartype(ContinuousVariable())
            for i, (attr, op, values) in enumerate(value):
                if context.attributes.get(attr) == CONTINUOUS:
                    if values and isinstance(values[0], str):
                        values = [QLocale().toDouble(v)[0] for v in values]
                        value[i] = (attr, op, values)
        return super().encode_setting(context, setting, value)

    def decode_setting(self, setting, value, domain=None):
        value = super().decode_setting(setting, value, domain)
        if setting.name == 'conditions':
            for i, (attr, op, values) in enumerate(value):
                var = attr in domain and domain[attr]
                if var and var.is_continuous and not isinstance(var, TimeVariable):
                    value[i] = (attr, op,
                                list([QLocale().toString(float(i), 'f')
                                      for i in values]))
        return value


class FilterDiscreteType(enum.Enum):
    Equal = "Equal"
    NotEqual = "NotEqual"
    In = "In"
    IsDefined = "IsDefined"


class OWSelectRows(widget.OWWidget):
    name = "Select Rows"
    id = "Orange.widgets.data.file"
    description = "Select rows from the data based on values of variables."
    icon = "icons/SelectRows.svg"
    priority = 100
    category = "Data"

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

        boxes = gui.widgetBox(self.controlArea, orientation=QGridLayout())
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

        attr_combo = gui.OrangeComboBox(
            minimumContentsLength=12,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon)
        attr_combo.row = row
        for var in self._visible_variables(self.data.domain):
            attr_combo.addItem(*gui.attributeItem(var))
        attr_combo.setCurrentIndex(attr or 0)
        self.cond_list.setCellWidget(row, 0, attr_combo)

        index = QPersistentModelIndex(model.index(row, 3))
        temp_button = QPushButton('×', self, flat=True,
                                  styleSheet='* {font-size: 16pt; color: silver}'
                                             '*:hover {color: black}')
        temp_button.clicked.connect(lambda: self.remove_one(index.row()))
        self.cond_list.setCellWidget(row, 3, temp_button)

        self.remove_all_button.setDisabled(False)
        self.set_new_operators(attr_combo, attr is not None,
                               condition_type, condition_value)
        attr_combo.currentIndexChanged.connect(
            lambda _: self.set_new_operators(attr_combo, False))

        self.cond_list.resizeRowToContents(row)

    @staticmethod
    def _visible_variables(domain):
        """Generate variables in order they should be presented in in combos."""
        return filter_visible(chain(domain.class_vars,
                                    domain.metas,
                                    domain.attributes))

    def add_all(self):
        if self.cond_list.rowCount():
            Mb = QMessageBox
            if Mb.question(
                    self, "Remove existing filters",
                    "This will replace the existing filters with "
                    "filters for all variables.", Mb.Ok | Mb.Cancel) != Mb.Ok:
                return
            self.remove_all()
        domain = self.data.domain
        for i in range(len(domain.variables) + len(domain.metas)):
            self.add_row(i)

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
        self.cond_list.clear()
        self.cond_list.setRowCount(0)
        self.remove_all_button.setDisabled(True)

    def set_new_operators(self, attr_combo, adding_all,
                          selected_index=None, selected_values=None):
        oper_combo = QComboBox()
        oper_combo.row = attr_combo.row
        oper_combo.attr_combo = attr_combo
        var = self.data.domain[attr_combo.currentText()]
        oper_combo.addItems(self.operator_names[type(var)])
        oper_combo.setCurrentIndex(selected_index or 0)
        self.cond_list.setCellWidget(oper_combo.row, 1, oper_combo)
        self.set_new_values(oper_combo, adding_all, selected_values)
        oper_combo.currentIndexChanged.connect(
            lambda _: self.set_new_values(oper_combo, False))

    @staticmethod
    def _get_lineedit_contents(box):
        return [child.text() for child in getattr(box, "controls", [box])
                if isinstance(child, QLineEdit)]

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

        def add_datetime(contents):
            le = add_textual(contents)
            le.setValidator(QRegExpValidator(QRegExp(TimeVariable.REGEX)))
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

        if oper_combo.currentText() == "is defined":
            label = QLabel()
            label.var_type = vartype(var)
            self.cond_list.setCellWidget(oper_combo.row, 2, label)
        elif var.is_discrete:
            if oper_combo.currentText() == "is one of":
                if selected_values:
                    lc = [x for x in list(selected_values)]
                button = DropDownToolButton(self, var, lc)
                button.var_type = vartype(var)
                self.cond_list.setCellWidget(oper_combo.row, 2, button)
            else:
                combo = QComboBox()
                combo.addItems([""] + var.values)
                if lc[0]:
                    combo.setCurrentIndex(int(lc[0]))
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
            elif var.is_string:
                box.controls = [add_textual(lc[0])]
                if oper in [6, 7]:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(add_textual(lc[1]))
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
            self.commit()
            return
        self.data_desc = report.describe_data_brief(data)
        self.conditions = []
        try:
            self.openContext(data)
        except Exception:
            pass

        variables = list(self._visible_variables(self.data.domain))
        varnames = [v.name for v in variables]
        if self.conditions:
            for attr, cond_type, cond_value in self.conditions:
                if attr in varnames:
                    self.add_row(varnames.index(attr), cond_type, cond_value)
        elif variables:
            self.add_row()

        self.update_info(data, self.data_in_variables, "In: ")
        self.unconditional_commit()

    def conditions_changed(self):
        try:
            self.conditions = []
            self.conditions = [
                (self.cond_list.cellWidget(row, 0).currentText(),
                 self.cond_list.cellWidget(row, 1).currentIndex(),
                 self._get_value_contents(self.cond_list.cellWidget(row, 2)))
                for row in range(self.cond_list.rowCount())]
            if self.update_on_change and (
                    self.last_output_conditions is None or
                    self.last_output_conditions != self.conditions):
                self.commit()
        except AttributeError:
            # Attribute error appears if the signal is triggered when the
            # controls are being constructed
            pass

    def _values_to_floats(self, attr, values):
        if not len(values):
            return values
        if not all(values):
            return None
        if isinstance(attr, TimeVariable):
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

    def commit(self):
        matching_output = self.data
        non_matching_output = None
        annotated_output = None

        self.Error.clear()
        if self.data:
            domain = self.data.domain
            conditions = []
            for attr_name, oper_idx, values in self.conditions:
                attr_index = domain.index(attr_name)
                attr = domain[attr_index]
                operators = self.Operators[type(attr)]
                opertype, _ = operators[oper_idx]
                if attr.is_continuous:
                    try:
                        floats = self._values_to_floats(attr, values)
                    except ValueError as e:
                        self.Error.parsing_error(e.args[0])
                        return
                    if floats is None:
                        continue
                    filter = data_filter.FilterContinuous(
                        attr_index, opertype, *floats)
                elif attr.is_string:
                    filter = data_filter.FilterString(
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
                    filter = data_filter.FilterDiscrete(attr_index, f_values)
                conditions.append(filter)

            if conditions:
                self.filters = data_filter.Values(conditions)
                matching_output = self.filters(self.data)
                self.filters.negate = True
                non_matching_output = self.filters(self.data)

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

        if matching_output is not None and not len(matching_output):
            matching_output = None
        if non_matching_output is not None and not len(non_matching_output):
            non_matching_output = None
        if annotated_output is not None and not len(annotated_output):
            annotated_output = None

        self.Outputs.matching_data.send(matching_output)
        self.Outputs.unmatched_data.send(non_matching_output)
        self.Outputs.annotated_data.send(annotated_output)

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
                          sp(len(data.domain.variables) +
                             len(data.domain.metas)))
                        )

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
        domain = self.data.domain
        for attr_name, oper, values in self.conditions:
            attr_index = domain.index(attr_name)
            attr = domain[attr_index]
            names = self.operator_names[type(attr)]
            name = names[oper]
            if oper == len(names) - 1:
                conditions.append("{} {}".format(attr, name))
            elif attr.is_discrete:
                if name == "is one of":
                    if len(values) == 1:
                        conditions.append("{} is {}".format(
                            attr, attr.values[values[0] - 1]))
                    elif len(values) > 1:
                        conditions.append("{} is {} or {}".format(
                            attr,
                            ", ".join(attr.values[v - 1] for v in values[:-1]),
                            attr.values[values[-1] - 1]))
                else:
                    if not (values and values[0]):
                        continue
                    value = values[0] - 1
                    conditions.append("{} {} {}".
                                      format(attr, name, attr.values[value]))
            else:
                if len(values) == 1:
                    conditions.append("{} {} {}".
                                      format(attr, name, *values))
                else:
                    conditions.append("{} {} {} and {}".
                                      format(attr, name, *values))
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


class CheckBoxPopup(QWidget):
    def __init__(self, var, lc, widget_parent=None, widget=None):
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

        self.widget = widget
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

    def resizeEvent(self, QResizeEvent):
        self.set_text()


def main(argv=None):  # pragma: no cover
    from AnyQt.QtWidgets import QApplication
    app = QApplication(argv or [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "zoo"

    w = OWSelectRows()
    w.set_data(Table(filename))
    w.show()
    app.exec_()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
