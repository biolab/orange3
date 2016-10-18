import sys
import numpy as np

from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QComboBox

import Orange.data
from Orange.preprocess import impute
from Orange.base import Learner
from Orange.widgets import gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget
from Orange.classification import SimpleTreeLearner


class DisplayFormatDelegate(QtGui.QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        method = index.data(Qt.UserRole)
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


class OWImpute(OWWidget):
    name = "Impute"
    description = "Impute missing values in the data table."
    icon = "icons/Impute.svg"
    priority = 2130

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Learner", Learner, "set_learner")]
    outputs = [("Data", Orange.data.Table)]

    DEFAULT_LEARNER = SimpleTreeLearner()
    METHODS = [AsDefault(), impute.DoNotImpute(), impute.Average(),
               impute.AsValue(), impute.Model(DEFAULT_LEARNER), impute.Random(),
               impute.DropInstances(), impute.Default()]
    DEFAULT, DO_NOT_IMPUTE, MODEL_BASED_IMPUTER, AS_INPUT = 0, 1, 4, 7

    settingsHandler = settings.DomainContextHandler()

    _default_method_index = settings.Setting(DO_NOT_IMPUTE)
    variable_methods = settings.ContextSetting({})
    autocommit = settings.Setting(False)
    default_value = settings.Setting(0.)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        main_layout = QtGui.QVBoxLayout()
        main_layout.setMargin(10)
        self.controlArea.layout().addLayout(main_layout)

        box = QtGui.QGroupBox(title=self.tr("Default Method"), flat=False)
        box_layout = QtGui.QVBoxLayout(box)
        main_layout.addWidget(box)

        button_group = QtGui.QButtonGroup()
        button_group.buttonClicked[int].connect(self.set_default_method)
        for i, method in enumerate(self.METHODS):
            if not method.columns_only:
                button = QtGui.QRadioButton(method.name)
                button.setChecked(i == self.default_method_index)
                button_group.addButton(button, i)
                box_layout.addWidget(button)

        self.default_button_group = button_group

        box = QtGui.QGroupBox(title=self.tr("Individual Attribute Settings"),
                              flat=False)
        main_layout.addWidget(box)

        horizontal_layout = QtGui.QHBoxLayout(box)
        main_layout.addWidget(box)

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

        horizontal_layout.addWidget(self.varview)

        method_layout = QtGui.QVBoxLayout()
        horizontal_layout.addLayout(method_layout)

        button_group = QtGui.QButtonGroup()
        for i, method in enumerate(self.METHODS):
            button = QtGui.QRadioButton(text=method.name)
            button_group.addButton(button, i)
            method_layout.addWidget(button)

        self.value_combo = QComboBox(
            minimumContentsLength=8,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            activated=self._on_value_selected
            )
        self.value_combo.currentIndexChanged.connect(self._on_value_changed)
        self.value_double = QtGui.QDoubleSpinBox(
            editingFinished=self._on_value_selected,
            minimum=-1000., maximum=1000., singleStep=.1, decimals=3,
            value=self.default_value
            )
        self.value_stack = value_stack = QtGui.QStackedLayout()
        value_stack.addWidget(self.value_combo)
        value_stack.addWidget(self.value_double)
        method_layout.addLayout(value_stack)

        button_group.buttonClicked[int].connect(
            self.set_method_for_current_selection
        )

        method_layout.addStretch(2)

        reset_button = QtGui.QPushButton(
                "Restore All to Default", checked=False, checkable=False,
                clicked=self.reset_variable_methods, default=False,
                autoDefault=False)
        method_layout.addWidget(reset_button)

        self.variable_button_group = button_group

        box = gui.auto_commit(
            self.controlArea, self, "autocommit", "Apply",
            orientation=Qt.Horizontal, checkbox_label="Apply automatically")
        box.layout().insertSpacing(0, 80)
        box.layout().insertWidget(0, self.report_button)

        self.data = None
        self.modified = False
        self.default_method = self.METHODS[self.default_method_index]
        self.update_varview()

    @property
    def default_method_index(self):
        return self._default_method_index

    @default_method_index.setter
    def default_method_index(self, index):
        if self._default_method_index != index:
            self._default_method_index = index
            self.default_button_group.button(index).setChecked(True)
            self.default_method = self.METHODS[self.default_method_index]
            self.METHODS[self.DEFAULT].method = self.default_method

            # update variable view
            for index in map(self.varmodel.index, range(len(self.varmodel))):
                self.varmodel.setData(index,
                                      self.variable_methods.get(index.row(), self.METHODS[self.DEFAULT]),
                                      Qt.UserRole)
            self._invalidate()

    def set_default_method(self, index):
        """Set the current selected default imputation method.
        """
        self.default_method_index = index

    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.varmodel[:] = []
        self.variable_methods = {}
        self.modified = False
        self.data = data

        if data is not None:
            self.varmodel[:] = data.domain.variables
            self.openContext(data.domain)

        self.update_varview()
        self.unconditional_commit()

    def set_learner(self, learner):
        self.learner = learner or self.DEFAULT_LEARNER
        imputer = self.METHODS[self.MODEL_BASED_IMPUTER]
        imputer.learner = self.learner

        button = self.default_button_group.button(self.MODEL_BASED_IMPUTER)
        button.setText(imputer.name)

        variable_button = self.variable_button_group.button(self.MODEL_BASED_IMPUTER)
        variable_button.setText(imputer.name)

        if learner is not None:
            self.default_method_index = self.MODEL_BASED_IMPUTER

        self.commit()

    def get_method_for_column(self, column_index):
        """Returns the imputation method for column by its index.
        """
        if not isinstance(column_index, int):
            column_index = column_index.row()

        return self.variable_methods.get(column_index,
                                         self.METHODS[self.DEFAULT])

    def _invalidate(self):
        self.modified = True
        self.commit()

    def commit(self):
        data = self.data

        if self.data is not None:
            drop_mask = np.zeros(len(self.data), bool)

            attributes = []
            class_vars = []

            self.warning()
            with self.progressBar(len(self.varmodel)) as progress:
                for i, var in enumerate(self.varmodel):
                    method = self.variable_methods.get(i, self.default_method)

                    if not method.supports_variable(var):
                        self.warning("Default method can not handle '{}'".
                                     format(var.name))
                    elif isinstance(method, impute.DropInstances):
                        drop_mask |= method(self.data, var)
                    else:
                        var = method(self.data, var)

                    if isinstance(var, Orange.data.Variable):
                        var = [var]

                    if i < len(self.data.domain.attributes):
                        attributes.extend(var)
                    else:
                        class_vars.extend(var)

                    progress.advance()

            domain = Orange.data.Domain(attributes, class_vars,
                                        self.data.domain.metas)
            data = self.data.from_table(domain, self.data[~drop_mask])

        self.send("Data", data)
        self.modified = False

    def send_report(self):
        specific = []
        for i, var in enumerate(self.varmodel):
            method = self.variable_methods.get(i, None)
            if method is not None:
                specific.append("{} ({})".format(var.name, str(method)))

        default = self.default_method.name
        if specific:
            self.report_items((
                ("Default method", default),
                ("Specific imputers", ", ".join(specific))
            ))
        else:
            self.report_items((("Method", default),))

    def _on_var_selection_changed(self):
        indexes = self.selection.selectedIndexes()
        methods = set(self.get_method_for_column(i.row()).name for i in indexes)

        selected_vars = [self.varmodel[index.row()] for index in indexes]
        has_discrete = any(var.is_discrete for var in selected_vars)

        if len(methods) == 1:
            method = methods.pop()
            for i, m in enumerate(self.METHODS):
                if method == m.name:
                    self.variable_button_group.button(i).setChecked(True)
        elif self.variable_button_group.checkedButton() is not None:
            self.variable_button_group.setExclusive(False)
            self.variable_button_group.checkedButton().setChecked(False)
            self.variable_button_group.setExclusive(True)

        for method, button in zip(self.METHODS,
                                  self.variable_button_group.buttons()):
            enabled = all(method.supports_variable(var) for var in
                          selected_vars)
            button.setEnabled(enabled)

        if not has_discrete:
            self.value_stack.setEnabled(True)
            self.value_stack.setCurrentWidget(self.value_double)
            self._on_value_changed()
        elif len(selected_vars) == 1:
            self.value_stack.setEnabled(True)
            self.value_stack.setCurrentWidget(self.value_combo)
            self.value_combo.clear()
            self.value_combo.addItems(selected_vars[0].values)
            self._on_value_changed()
        else:
            self.variable_button_group.button(self.AS_INPUT).setEnabled(False)
            self.value_stack.setEnabled(False)

    def set_method_for_current_selection(self, method_index):
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, method_index)

    def set_method_for_indexes(self, indexes, method_index):
        if method_index == self.DEFAULT:
            for index in indexes:
                self.variable_methods.pop(index, None)
        else:
            method = self.METHODS[method_index].copy()
            for index in indexes:
                self.variable_methods[index.row()] = method

        self.update_varview(indexes)
        self._invalidate()

    def update_varview(self, indexes=None):
        if indexes is None:
            indexes = map(self.varmodel.index, range(len(self.varmodel)))

        for index in indexes:
            self.varmodel.setData(index, self.get_method_for_column(index.row()), Qt.UserRole)

    def _on_value_selected(self):
        self.variable_button_group.button(self.AS_INPUT).setChecked(True)
        self._on_value_changed()

    def _on_value_changed(self):
        widget = self.value_stack.currentWidget()
        if widget is self.value_combo:
            value = self.value_combo.currentText()
        else:
            value = self.value_double.value()
            self.default_value = value

        self.METHODS[self.AS_INPUT].default = value
        index = self.variable_button_group.checkedId()
        if index == self.AS_INPUT:
            self.set_method_for_current_selection(index)

    def reset_variable_methods(self):
        indexes = map(self.varmodel.index, range(len(self.varmodel)))
        self.set_method_for_indexes(indexes, self.DEFAULT)
        self.variable_button_group.button(self.DEFAULT).setChecked(True)


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
    app.exec_()
    w.set_data(None)
    w.set_learner(None)
    w.handleNewSignals()
    w.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main())
