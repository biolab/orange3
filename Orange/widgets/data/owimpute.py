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


class DisplayFormatDelegate(QtGui.QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        method = index.data(Qt.UserRole)
        var = index.model()[index.row()]
        if method:
            option.text = method.str(var)
            if (not method.support_continuous and var.is_continuous) or \
               (not method.support_discrete and var.is_discrete):
                option.palette.setColor(option.palette.Text, Qt.gray)


class Default(impute.BaseImputeMethod):
    name = "Default (above)"
    short_name = ""
    format = "{var.name}"
    columns_only = True

    method = impute.BaseImputeMethod()

    @property
    def support_discrete(self):
        return self.method.support_discrete

    @property
    def support_continuous(self):
        return self.method.support_continuous


class OWImpute(OWWidget):
    name = "Impute"
    description = "Impute missing values in the data table."
    icon = "icons/Impute.svg"
    priority = 2130

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Learner", Learner, "set_learner")]
    outputs = [("Data", Orange.data.Table)]

    METHODS = [Default(), impute.BaseImputeMethod(), impute.Average(),
               impute.AsValue(), impute.Model(), impute.Random(),
               impute.DropInstances(), impute.Default()]
    DEFAULT, DONT_IMPUTE, MODEL_BASED_IMPUTER, AS_DEFAULT = 0, 1, 4, 7

    settingsHandler = settings.DomainContextHandler()

    _default_method_index = settings.Setting(DONT_IMPUTE)
    variable_methods = settings.ContextSetting({})
    autocommit = settings.Setting(False)
    value_line = settings.Setting('value')

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        main_layout = QtGui.QVBoxLayout()
        main_layout.setMargin(10)
        self.controlArea.layout().addLayout(main_layout)

        box = QtGui.QGroupBox(title=self.tr("Default method"), flat=False)
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

        box = QtGui.QGroupBox(title=self.tr("Individual attribute settings"),
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
            activated=self._on_value_changed)
        self.value_double = QtGui.QDoubleSpinBox(
            editingFinished=self._on_value_changed,
            minimum=-1000., maximum=1000., singleStep=.1, decimals=3
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
                "Restore all to default", checked=False, checkable=False,
                clicked=self.reset_variable_methods, default=False,
                autoDefault=False)
        method_layout.addWidget(reset_button)

        self.variable_button_group = button_group

        box = gui.auto_commit(
            self.controlArea, self, "autocommit", "Commit",
            orientation=Qt.Horizontal, checkbox_label="Commit on any change")
        box.layout().insertSpacing(0, 80)
        box.layout().insertWidget(0, self.report_button)

        self.data = None
        self.modified = False
        self.default_method = self.METHODS[self.default_method_index]
        self.set_learner(None)

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
            itemmodels.select_row(self.varview, 0)

        self.unconditional_commit()

    def set_learner(self, learner):
        self.learner = learner
        self.METHODS[self.MODEL_BASED_IMPUTER].learner = learner

        button = self.default_button_group.button(self.MODEL_BASED_IMPUTER)
        variable_button = self.variable_button_group.button(self.MODEL_BASED_IMPUTER)
        if self.learner is None:
            button.setEnabled(False)
            variable_button.setEnabled(False)
            self.set_default_method(self.DONT_IMPUTE)
        else:
            self.default_method_index = self.MODEL_BASED_IMPUTER
            button.setText("{} ({})".format(self.default_method.name,
                                            learner.name))
            button.setEnabled(True)
            variable_button.setEnabled(True)

        self.commit()

    def get_method_for_column(self, column_index):
        """Returns the imputation method for column by its index.
        """
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

            with self.progressBar(len(self.varmodel)) as progress:
                for i, var in enumerate(self.varmodel):
                    method = self.variable_methods.get(i, self.default_method)

                    if isinstance(method, impute.DropInstances):
                        drop_mask |= method(self.data, var)
                    else:
                        var = method(self.data, var)

                    if i < len(self.data.domain.attributes):
                        attributes.append(var)
                    else:
                        class_vars.append(var)

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
        has_continuous = any(var.is_continuous for var in selected_vars)
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
            enabled = (not has_continuous or method.support_continuous) and \
                      (not has_discrete or method.support_discrete)
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
            self.variable_button_group.button(self.AS_DEFAULT).setEnabled(False)
            self.value_stack.setEnabled(False)

    def set_method_for_current_selection(self, method_index):
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, method_index)

    def set_method_for_indexes(self, indexes, method_index):
        method = self.METHODS[method_index].copy()

        for index in indexes:
            self.varmodel.setData(index, method, Qt.UserRole)
            self.variable_methods[index.row()] = method

        self._invalidate()

    def _on_value_changed(self):
        widget = self.value_stack.currentWidget()
        if widget is self.value_combo:
            value = self.value_combo.currentText()
        else:
            value = self.value_double.value()

        self.METHODS[self.AS_DEFAULT].default = value
        index = self.variable_button_group.checkedId()
        self.set_method_for_current_selection(index)

    def reset_variable_methods(self):
        indexes = map(self.varmodel.index, range(len(self.varmodel)))
        self.set_method_for_indexes(indexes, self.DEFAULT)
        self.variable_button_group.button(self.DEFAULT).setChecked(True)


def main(argv=sys.argv):
    from Orange.classification import SimpleTreeLearner
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
    w.set_learner(SimpleTreeLearner())
    w.handleNewSignals()
    app.exec_()
    w.set_data(None)
    w.set_learner(None)
    w.handleNewSignals()
    w.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main())
