import sys
from functools import wraps

from collections import namedtuple

from PyQt4 import QtGui
from PyQt4.QtGui import (
    QWidget, QGroupBox, QRadioButton, QPushButton, QHBoxLayout,
    QVBoxLayout, QStackedLayout, QComboBox, QLineEdit,
    QDoubleValidator, QButtonGroup
)
from PyQt4.QtCore import Qt, QMargins

import Orange.data
import Orange.preprocess.impute as impute
import Orange.classification

from Orange.base import Learner

from Orange.widgets import gui, settings
from Orange.widgets.utils import itemmodels, vartype
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget


def _margins(margins, container):
    if isinstance(margins, tuple):
        left, top, right, bottom = margins
    elif isinstance(margins, int):
        left = top = right = bottom = margins
    elif isinstance(margins, QMargins):
        left, top, right, bottom = \
            margins.left(), margins.top(), margins.right(), margins.bottom()
    else:
        raise TypeError

    container_margins = container.getContentsMargins()
    margins = [c if m == -1 else m
               for c, m in zip([left, top, right, bottom],
                               container_margins)]
    return margins


def layout(orientation=Qt.Vertical, margins=None, spacing=None,):
    if orientation == Qt.Vertical:
        lay = QVBoxLayout()
    else:
        lay = QHBoxLayout()

    if margins is not None:
        left, top, right, bottom = _margins(margins, lay)
        lay.setContentsMargins(left, right, top, bottom)
    return lay


def group_box(title=None, layout=None, margin=None, flat=False, ):
    gb = QGroupBox(title=title, flat=flat)
    if layout is not None:
        gb.setLayout(layout)
    return gb


def widget(layout=None, tooltip=None, objname=None, enabled=True,):
    w = QWidget(toolTip=tooltip, objectName=objname, enabled=enabled)
    if layout is not None:
        w.setLayout(layout)
    return w


def radio_button(text="", checked=False, group=None, group_id=None):
    button = QRadioButton(text, checked=checked)
    if group is not None:
        group.addButton(button, )
        if group_id is not None:
            group.setId(button, group_id)
    return button


def push_button(text="", checked=False, checkable=False,
                group=None, group_id=None, **kwargs):
    button = QPushButton(text, checked=checked, checkable=checkable, **kwargs)
    if group is not None:
        group.addButton(button)
        if group_id is not None:
            group.setId(button, group_id)
    return button


class DisplayFormatDelegate(QtGui.QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        state = index.data(Qt.UserRole)
        var = index.model()[index.row()]
        if state:
            fmt = state.method.format
            text = fmt.format(var=var, params=state.params,
                              **state.method._asdict())

            if var in show_error_vars:
                text += " (!)"

            option.text = text

show_error_vars = set()

METHODS = (
    {"name": "Default (above)",
     "short": "",
     "description": "As above so below",
     "format": "{var.name}"},
    {"name": "Don't impute",
     "short": "leave",
     "description": "I",
     "format": "{var.name} -> leave"},
    {"name": "Average/Most frequent",
     "short": "avg",
     "description": "Replace with average/modus for the column",
     "format": "{var.name} -> avg"},
    {"name": "As a distinct value",
     "short": "as_value",
     "description": "",
     "format": "{var.name} -> new value"},
    {"name": "Model-based imputer",
     "short": "model",
     "description": "",
     "format": "{var.name} -> {params[0]!r}"},
    {"name": "Random values",
     "short": "random",
     "description": "Replace with a random value",
     "format": "{var.name} -> random"},
    {"name": "Remove instances with unknown values",
     "short": "drop",
     "description": "",
     "format": "{var.name} -> drop"},
    {"name": "Value",
     "short": "value",
     "description": "",
     "format": "{var.name} -> {params[0]!s}"},
)


Method = namedtuple(
    "Method",
    ["name", "short", "description", "format"]
)


class Method(Method):
    pass


State = namedtuple("State", ["method", "params"])


class State(State):
    def __new__(cls, method, params=()):
        return super().__new__(cls, method, params)

    def _asdict(self):
        return {"method": self.method._asdict(),
                "params": self.params}

# state
#  - selected default
#  - for each variable (indexed by (vartype, name)):
#     - selected method (method index, *params)

# vartype * name -> method
# data method = Method(name) | Method2(name, (*params))


METHODS = [Method(**m) for m in METHODS]


class OWImpute(OWWidget):
    name = "Impute"
    description = "Impute missing values in the data table."
    icon = "icons/Impute.svg"
    priority = 2130

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Learner", Learner, "set_learner")]
    outputs = [("Data", Orange.data.Table)]

    METHODS = METHODS
    model_mthd_index = [i for i, m in enumerate(METHODS[1:-1], 1) if
                        m.short == "model"][0]

    settingsHandler = settings.DomainContextHandler()

    default_method = settings.Setting(1)
    variable_methods = settings.ContextSetting({})
    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.modified = False

        box = group_box(self.tr("Default method"),
                        layout=layout(Qt.Vertical))
        self.controlArea.layout().addWidget(box)

        bgroup = QButtonGroup()

        for i, m in enumerate(self.METHODS[1:-1], 1):
            b = radio_button(m.name, checked=i == self.default_method,
                             group=bgroup, group_id=i)
            box.layout().addWidget(b)

        self.defbggroup = bgroup

        bgroup.buttonClicked[int].connect(self.set_default_method)
        box = group_box(self.tr("Individual attribute settings"),
                        layout=layout(Qt.Horizontal))
        self.controlArea.layout().addWidget(box)

        self.varview = QtGui.QListView(
            selectionMode=QtGui.QListView.ExtendedSelection
        )
        self.varview.setItemDelegate(DisplayFormatDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._on_var_selection_changed
        )
        self.selection = self.varview.selectionModel()

        box.layout().addWidget(self.varview)

        method_layout = layout(Qt.Vertical, margins=0)
        box.layout().addLayout(method_layout)

        methodbox = group_box(layout=layout(Qt.Vertical))

        bgroup = QButtonGroup()
        for i, m in enumerate(self.METHODS):
            b = radio_button(m.name, group=bgroup, group_id=i)
            methodbox.layout().addWidget(b)

        assert self.METHODS[-1].short == "value"

        self.value_stack = value_stack = QStackedLayout()
        self.value_combo = QComboBox(
            minimumContentsLength=8,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            activated=self._on_value_changed)
        self.value_line = QLineEdit(editingFinished=self._on_value_changed)
        self.value_line.setValidator(QDoubleValidator())
        value_stack.addWidget(self.value_combo)
        value_stack.addWidget(self.value_line)
        methodbox.layout().addLayout(value_stack)

        bgroup.buttonClicked[int].connect(
            self.set_method_for_current_selection
        )
        reset_button = push_button("Restore all to default",
                                   clicked=self.reset_var_methods,
                                   default=False, autoDefault=False)

        method_layout.addWidget(methodbox)
        method_layout.addStretch(2)
        method_layout.addWidget(reset_button)
        self.varmethodbox = methodbox
        self.varbgroup = bgroup

        box = gui.auto_commit(
            self.controlArea, self, "autocommit", "Commit",
            orientation=Qt.Horizontal, checkbox_label="Commit on any change")
        box.layout().insertSpacing(0, 80)
        box.layout().insertWidget(0, self.report_button)
        self.data = None
        self.learner = None
        self.not_compatible_with_learner = None

    def set_default_method(self, index):
        """
        Set the current selected default imputation method.
        """
        if self.default_method != index:
            self.default_method = index
            self.defbggroup.button(index).setChecked(True)
            self._invalidate()

    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.data = data
        if data is not None:
            self.varmodel[:] = data.domain.variables
            self.openContext(data.domain)
            self.restore_state(self.variable_methods)
            itemmodels.select_row(self.varview, 0)
        self.check_var_compatibility_with_learner()
        self.unconditional_commit()

    def set_learner(self, learner):
        self.learner = learner
        self.check_var_compatibility_with_learner()
        if self.model_mthd_index == self.default_method:
            self._invalidate()
        else:
            self.set_default_method(self.model_mthd_index)

    def check_var_compatibility_with_learner(self):
        data = self.data
        self.not_compatible_with_learner = set()

        if data is not None and self.learner is not None:
            for var in self.varmodel:
                domain = impute.domain_with_class_var(data.domain, var)
                if not self.learner.check_learner_adequacy(domain):
                    self.not_compatible_with_learner.add(var)

    def restore_state(self, state):
        for i, var in enumerate(self.varmodel):
            key = variable_key(var)
            if key in state:
                index = self.varmodel.index(i)
                self.varmodel.setData(index, state[key], Qt.UserRole)

    def clear(self):
        self.varmodel[:] = []
        self.variable_methods = {}
        self.data = None
        self.modified = False

    def state_for_column(self, column):
        """
        #:: int -> State
        Return the effective imputation state for `column`.

        :param int column:
        :rtype State:

        """
        var = self.varmodel[column]

        state = self.variable_methods.get(variable_key(var), None)
        if state is None or state.method == METHODS[0]:
            state = State(METHODS[self.default_method], ())
        return state

    def imputed_vars_for_column(self, column):
        state = self.state_for_column(column)
        data = self.data
        var = data.domain[column]
        method, params = state
        if method.short == "leave":
            return var
        elif method.short == "drop":
            return var
        elif method.short == "avg":
            return impute.Average()(data, var)
        elif method.short == "model":
            learner = (self.learner if self.learner is not None
                       else Orange.classification.SimpleTreeLearner())
            return impute.Model(learner)(data, var)
        elif method.short == "random":
            return impute.Random()(data, var)
        elif method.short == "value":
            return impute.Default(float(params[0]))(data, var)
        elif method.short == "as_value":
            return impute.AsValue()(data, var)
        else:
            assert False

    def _handle_learner_inadequacy(func):
        """
        Indicate an error on learner inadequacy for given data domain
        variable(s) when model-based imputation is selected.

        To avoid raising exceptions and consequently crashing the widget,
        commit() calls are deterred until a suitable input learner
        configuration is selected.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            global show_error_vars
            show_error_vars = set()
            for i, var in enumerate(self.varmodel):
                state = self.state_for_column(i)
                if (var in self.not_compatible_with_learner and
                        state.method.short == "model"):
                    show_error_vars.add(var)

            if not show_error_vars:
                self.error()
                func(self, *args, **kwargs)
            else:
                self.error(self.learner.learner_adequacy_err_msg)
        return wrapper

    @_handle_learner_inadequacy
    def commit(self):
        if self.data is not None:
            varstates = [(var, self.state_for_column(i))
                         for i, var in enumerate(self.varmodel)]
            attrs = []
            class_vars = []
            filter_columns = []
            for i, (var, state) in enumerate(varstates):
                if state.method.short == "drop":
                    imputedvars = [var]
                    filter_columns.append(i)
                elif state.method.short == "leave":
                    imputedvars = [var]
                else:
                    imputedvars = self.imputed_vars_for_column(i)
                    if imputedvars is None:
                        imputedvars = []
                    elif isinstance(imputedvars, Orange.data.Variable):
                        imputedvars = [imputedvars]

                if i < len(self.data.domain.attributes):
                    attrs.extend(imputedvars)
                else:
                    class_vars.extend(imputedvars)

            domain = Orange.data.Domain(
                attrs, class_vars, self.data.domain.metas)

            data = self.data.from_table(domain, self.data)

            if filter_columns:
                filter_ = Orange.data.filter.IsDefined(filter_columns)
                data = filter_(data)
        else:
            data = None

        self.send("Data", data)
        self.modified = False

    def send_report(self):
        specific = []
        for var in self.varmodel:
            state = self.variable_methods.get(variable_key(var), None)
            if state is not None and state.method.short:
                if state.method.short == "value":
                    if var.is_continuous:
                        specific.append(
                            "{} (impute value {})".
                            format(var.name, float(state.params[0])))
                    else:
                        specific.append(
                            "{} (impute value '{}'".
                            format(var.name, var.values[state.params[0]]))
                else:
                    specific.append(
                        "{} ({})".
                        format(var.name, state.method.name.lower()))
        default = self.METHODS[self.default_method].name
        if specific:
            self.report_items((
                ("Default method", default),
                ("Specific imputers", ", ".join(specific))
            ))
        else:
            self.report_items((("Method", default),))

    def _invalidate(self):
        self.modified = True
        self.commit()

    def _on_var_selection_changed(self):
        indexes = self.selection.selectedIndexes()

        vars = [self.varmodel[index.row()] for index in indexes]
        defstate = State(METHODS[0], ())
        states = [self.variable_methods.get(variable_key(var), defstate)
                  for var in vars]
        all_cont = all(var.is_continuous for var in vars)
        states = list(unique(states))
        method = None
        params = ()
        state = None
        if len(states) == 1:
            state = states[0]
            method, params = state
            mindex = METHODS.index(method)
            self.varbgroup.button(mindex).setChecked(True)
        elif self.varbgroup.checkedButton() is not None:
            self.varbgroup.setExclusive(False)
            self.varbgroup.checkedButton().setChecked(False)
            self.varbgroup.setExclusive(True)

        values, enabled, stack_index = [], False, 0
        value, value_index = "0.0", 0
        if all_cont:
            enabled, stack_index = True, 1
            if method is not None and method.short == "value":
                value = params[0]

        elif len(vars) == 1 and vars[0].is_discrete:
            values, enabled, stack_index = vars[0].values, True, 0
            if method is not None and method.short == "value":
                try:
                    value_index = values.index(params[0])
                except IndexError:
                    pass

        self.value_stack.setCurrentIndex(stack_index)
        self.value_stack.setEnabled(enabled)

        if stack_index == 0:
            self.value_combo.clear()
            self.value_combo.addItems(values)
            self.value_combo.setCurrentIndex(value_index)
        else:
            self.value_line.setText(value)

    def _on_value_changed(self):
        # The "fixed" value in the widget has been changed by the user.
        index = self.varbgroup.checkedId()
        self.set_method_for_current_selection(index)

    def set_method_for_current_selection(self, methodindex):
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, methodindex)

    def set_method_for_indexes(self, indexes, methodindex):
        method = METHODS[methodindex]
        params = (None,)
        if method.short == "value":
            if self.value_stack.currentIndex() == 0:
                value = self.value_combo.currentIndex()
            else:
                value = self.value_line.text()
            params = (value, )
        elif method.short == "model":
            params = ("model", )
        state = State(method, params)

        for index in indexes:
            self.varmodel.setData(index, state, Qt.UserRole)
            var = self.varmodel[index.row()]
            self.variable_methods[variable_key(var)] = state

        self._invalidate()

    def reset_var_methods(self):
        indexes = map(self.varmodel.index, range(len(self.varmodel)))
        self.set_method_for_indexes(indexes, 0)


def variable_key(variable):
    return (vartype(variable), variable.name)


def unique(iterable):
    seen = set()
    for el in iterable:
        if el not in seen:
            seen.add(el)
            yield el


def main(argv=sys.argv):
    app = QtGui.QApplication(list(argv))
    argv = app.argv()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    w = OWImpute()
    w.show()
    w.raise_()

    data = Orange.data.Table(filename)
    w.set_data(data)
    w.handleNewSignals()
    w.set_learner(Orange.classification.SimpleTreeLearner())
    w.handleNewSignals()
    app.exec_()
    w.set_data(None)
    w.set_learner(None)
    w.handleNewSignals()
    w.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main())
