import enum
from collections import OrderedDict
from itertools import chain

import numpy as np

from AnyQt.QtWidgets import (
    QWidget, QTableWidget, QHeaderView, QComboBox, QLineEdit, QToolButton,
    QMessageBox, QMenu, QListView, QGridLayout, QPushButton, QSizePolicy,
    QLabel, QHBoxLayout)
from AnyQt.QtGui import (
    QDoubleValidator, QRegExpValidator, QStandardItemModel, QStandardItem,
    QFontMetrics, QPalette
)
from AnyQt.QtCore import Qt, QPoint, QRegExp, QPersistentModelIndex, QLocale
from orangewidget.utils.combobox import ComboBoxSearch

from Orange.data import (
    Variable, ContinuousVariable, DiscreteVariable, StringVariable,
    TimeVariable,
    Table
)
import Orange.data.filter as data_filter
from Orange.data.filter import FilterContinuous, FilterString
from Orange.data.domain import filter_visible
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
from Orange.widgets.utils.state_summary import format_summary_details


class SelectRowsContextHandler(DomainContextHandler):
    """Context handler that filters conditions"""

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
            vtype = context.attributes.get(attr)
            if vtype == CONTINUOUS and values and isinstance(values[0], str):
                values = [QLocale().toDouble(v)[0] for v in values]
            encoded.append((attr, vtype, op, values))
        return encoded

    def decode_setting(self, setting, value, domain=None):
        value = super().decode_setting(setting, value, domain)
        if setting.name == 'conditions':
            # Use this after 2022/2/2:
            # for i, (attr, _, op, values) in enumerate(value):
            for i, condition in enumerate(value):
                attr = condition[0]
                op, values = condition[-2:]

                var = attr in domain and domain[attr]
                if var and var.is_continuous and not isinstance(var, TimeVariable):
                    values = [QLocale().toString(float(i), 'f') for i in values]
                value[i] = (attr, op, values)
        return value

    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return self.PERFECT_MATCH

        conditions = context.values["conditions"]
        all_vars = attrs.copy()
        all_vars.update(metas)
        # Use this after 2022/2/2:
        # if all(all_vars.get(name) == tpe for name, tpe, *_ in conditions):
        if all(all_vars.get(name) == tpe if len(rest) == 2 else name in all_vars
               for name, tpe, *rest in conditions):
            return 0.5
        return self.NO_MATCH


class FilterDiscreteType(enum.Enum):
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
    id = "Orange.widgets.data.file"
    description = "Select rows from the data based on values of variables."
    icon = "icons/SelectRows.svg"
    priority = 100
    category = "Data"
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

        boxes = gui.widgetBox(self.controlArea, orientation=QHBoxLayout())
        layout = boxes.layout()

        box_setting = gui.vBox(boxes, addToLayout=False, box=True)
        self.cb_pa = gui.checkBox(
            box_setting, self, "purge_attributes", "Remove unused features",
            callback=self.conditions_changed)
        gui.separator(box_setting, height=1)
        self.cb_pc = gui.checkBox(
            box_setting, self, "purge_classes", "Remove unused classes",
            callback=self.conditions_changed)
        layout.addWidget(box_setting, 1)

        self.report_button.setFixedWidth(120)
        gui.rubber(self.buttonsArea.layout())
        layout.addWidget(self.buttonsArea)

        acbox = gui.auto_send(None, self, "auto_commit")
        layout.addWidget(acbox, 1)
        layout.setAlignment(acbox, Qt.AlignBottom)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

        self.set_data(None)
        self.resize(600, 400)

    def add_row(self, attr=None, condition_type=None, condition_value=None):
        model = self.cond_list.model()
        row = model.rowCount()
        model.insertRow(row)

        attr_combo = ComboBoxSearch(
            minimumContentsLength=12,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon)
        attr_combo.row = row
        for var in self._visible_variables(self.data.domain):
            if isinstance(var, Variable):
                attr_combo.addItem(*gui.attributeItem(var))
            else:
                attr_combo.addItem(var)
        if isinstance(attr, str):
            attr_combo.setCurrentText(attr)
        else:
            attr_combo.setCurrentIndex(
                attr or
                len(self.AllTypes) - (attr_combo.count() == len(self.AllTypes)))
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

    @classmethod
    def _visible_variables(cls, domain):
        """Generate variables in order they should be presented in in combos."""
        return chain(
            cls.AllTypes,
            filter_visible(chain(domain.class_vars,
                                 domain.metas,
                                 domain.attributes)))

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
        attr_name = attr_combo.currentText()
        if attr_name in self.AllTypes:
            oper_combo.addItems(self.operator_names[attr_name])
        else:
            var = self.data.domain[attr_name]
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

        box = self.cond_list.cellWidget(oper_combo.row, 2)
        lc = ["", ""]
        oper = oper_combo.currentIndex()
        attr_name = oper_combo.attr_combo.currentText()
        if attr_name in self.AllTypes:
            vtype = self.AllTypes[attr_name]
            var = None
        else:
            var = self.data.domain[attr_name]
            vtype = vartype(var)
            if selected_values is not None:
                lc = list(selected_values) + ["", ""]
                lc = [str(x) for x in lc[:2]]
        if box and vtype == box.var_type:
            lc = self._get_lineedit_contents(box) + lc

        if oper_combo.currentText().endswith(" defined"):
            label = QLabel()
            label.var_type = vtype
            self.cond_list.setCellWidget(oper_combo.row, 2, label)
        elif var is not None and var.is_discrete:
            if oper_combo.currentText().endswith(" one of"):
                if selected_values:
                    lc = [x for x in list(selected_values)]
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
            if vtype in (2, 4):  # continuous, time:
                validator = add_datetime if isinstance(var, TimeVariable) else add_numeric
                box.controls = [validator(lc[0])]
                if oper > 5:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(validator(lc[1]))
            elif vtype == 3:  # string:
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
            self.info.set_input_summary(self.info.NoInput)
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
        varnames = [v.name if isinstance(v, Variable) else v for v in variables]
        if self.conditions:
            for attr, cond_type, cond_value in self.conditions:
                if attr in varnames:
                    self.add_row(varnames.index(attr), cond_type, cond_value)
                elif attr in self.AllTypes:
                    self.add_row(attr, cond_type, cond_value)
        else:
            self.add_row()

        self.info.set_input_summary(len(data),
                                    format_summary_details(data))
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
                    filter = data_filter.IsDefined()
                elif attr_type in (2, 4):  # continuous, time
                    try:
                        floats = self._values_to_floats(attr, values)
                    except ValueError as e:
                        self.Error.parsing_error(e.args[0])
                        return
                    if floats is None:
                        continue
                    filter = data_filter.FilterContinuous(
                        attr_index, opertype, *floats)
                elif attr_type == 3:  # string
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

        summary = len(matching_output) if matching_output else self.info.NoOutput
        details = format_summary_details(matching_output) if matching_output else ""
        self.info.set_output_summary(summary, details)

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
            if attr_name in self.AllTypes:
                attr = attr_name
                names = self.operator_names[attr_name]
                var_type = self.AllTypes[attr_name]
            else:
                attr = domain[attr_name]
                var_type = vartype(attr)
                names = self.operator_names[type(attr)]
            name = names[oper]
            if oper == len(names) - 1:
                conditions.append("{} {}".format(attr, name))
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

    # Uncomment this on 2022/2/2
    #
    # @classmethod
    # def migrate_context(cls, context, version):
    #     if not version or version < 2:
    #         # Just remove; can't migrate because variables types are unknown
    #         context.values["conditions"] = []


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


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSelectRows).run(Table("zoo"))
