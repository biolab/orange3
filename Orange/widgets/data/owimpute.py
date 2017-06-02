import sys
import copy

import numpy as np

from AnyQt.QtWidgets import (
    QGroupBox, QRadioButton, QPushButton, QHBoxLayout,
    QVBoxLayout, QStackedWidget, QComboBox,
    QButtonGroup, QStyledItemDelegate, QListView, QDoubleSpinBox
)
from AnyQt.QtCore import Qt

import Orange.data
from Orange.preprocess import impute
from Orange.base import Learner
from Orange.widgets import gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Msg
from Orange.classification import SimpleTreeLearner


class DisplayFormatDelegate(QStyledItemDelegate):
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

    class Error(OWWidget.Error):
        imputation_failed = Msg("Imputation failed for '{}'")
        model_based_imputer_sparse = Msg("Model based imputer does not work for sparse data")

    DEFAULT_LEARNER = SimpleTreeLearner()
    METHODS = [AsDefault(), impute.DoNotImpute(), impute.Average(),
               impute.AsValue(), impute.Model(DEFAULT_LEARNER), impute.Random(),
               impute.DropInstances(), impute.Default()]
    DEFAULT, DO_NOT_IMPUTE, MODEL_BASED_IMPUTER, AS_INPUT = 0, 1, 4, 7

    settingsHandler = settings.DomainContextHandler()

    _default_method_index = settings.Setting(DO_NOT_IMPUTE)
    variable_methods = settings.ContextSetting({})
    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        # copy METHODS (some are modified by the widget)
        self.methods = copy.deepcopy(OWImpute.METHODS)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.controlArea.layout().addLayout(main_layout)

        box = QGroupBox(title=self.tr("Default Method"), flat=False)
        box_layout = QVBoxLayout(box)
        main_layout.addWidget(box)

        button_group = QButtonGroup()
        button_group.buttonClicked[int].connect(self.set_default_method)
        for i, method in enumerate(self.methods):
            if not method.columns_only:
                button = QRadioButton(method.name)
                button.setChecked(i == self.default_method_index)
                button_group.addButton(button, i)
                box_layout.addWidget(button)

        self.default_button_group = button_group

        box = QGroupBox(title=self.tr("Individual Attribute Settings"),
                        flat=False)
        main_layout.addWidget(box)

        horizontal_layout = QHBoxLayout(box)
        main_layout.addWidget(box)

        self.varview = QListView(
            selectionMode=QListView.ExtendedSelection
        )
        self.varview.setItemDelegate(DisplayFormatDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._on_var_selection_changed
        )
        self.selection = self.varview.selectionModel()

        horizontal_layout.addWidget(self.varview)

        method_layout = QVBoxLayout()
        horizontal_layout.addLayout(method_layout)

        button_group = QButtonGroup()
        for i, method in enumerate(self.methods):
            button = QRadioButton(text=method.name)
            button_group.addButton(button, i)
            method_layout.addWidget(button)

        self.value_combo = QComboBox(
            minimumContentsLength=8,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            activated=self._on_value_selected
            )
        self.value_double = QDoubleSpinBox(
            editingFinished=self._on_value_selected,
            minimum=-1000., maximum=1000., singleStep=.1, decimals=3,
            )
        self.value_stack = value_stack = QStackedWidget()
        value_stack.addWidget(self.value_combo)
        value_stack.addWidget(self.value_double)
        method_layout.addWidget(value_stack)

        button_group.buttonClicked[int].connect(
            self.set_method_for_current_selection
        )

        method_layout.addStretch(2)

        reset_button = QPushButton(
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
        self.learner = None
        self.modified = False
        self.default_method = self.methods[self.default_method_index]

    @property
    def default_method_index(self):
        return self._default_method_index

    @default_method_index.setter
    def default_method_index(self, index):
        if self._default_method_index != index:
            self._default_method_index = index
            self.default_button_group.button(index).setChecked(True)
            self.default_method = self.methods[self.default_method_index]
            self.methods[self.DEFAULT].method = self.default_method

            # update variable view
            for index in map(self.varmodel.index, range(len(self.varmodel))):
                method = self.variable_methods.get(
                    index.row(), self.methods[self.DEFAULT])
                self.varmodel.setData(index, method, Qt.UserRole)
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
        imputer = self.methods[self.MODEL_BASED_IMPUTER]
        imputer.learner = self.learner

        button = self.default_button_group.button(self.MODEL_BASED_IMPUTER)
        button.setText(imputer.name)

        variable_button = self.variable_button_group.button(self.MODEL_BASED_IMPUTER)
        variable_button.setText(imputer.name)

        if learner is not None:
            self.default_method_index = self.MODEL_BASED_IMPUTER

        self.update_varview()
        self.commit()

    def get_method_for_column(self, column_index):
        """Returns the imputation method for column by its index.
        """
        if not isinstance(column_index, int):
            column_index = column_index.row()

        return self.variable_methods.get(column_index,
                                         self.methods[self.DEFAULT])

    def _invalidate(self):
        self.modified = True
        self.commit()

    def commit(self):
        data = self.data

        if self.data is not None:
            if not len(self.data):
                self.send("Data", self.data)
                self.modified = False
                return

            drop_mask = np.zeros(len(self.data), bool)

            attributes = []
            class_vars = []

            self.warning()
            self.Error.imputation_failed.clear()
            self.Error.model_based_imputer_sparse.clear()
            with self.progressBar(len(self.varmodel)) as progress:
                for i, var in enumerate(self.varmodel):
                    method = self.variable_methods.get(i, self.default_method)
                    if isinstance(method, impute.Model) and data.is_sparse():
                        self.Error.model_based_imputer_sparse()
                        continue

                    try:
                        if not method.supports_variable(var):
                            self.warning("Default method can not handle '{}'".
                                         format(var.name))
                        elif isinstance(method, impute.DropInstances):
                                drop_mask |= method(self.data, var)
                        else:
                            var = method(self.data, var)
                    except Exception:  # pylint: disable=broad-except
                        self.Error.imputation_failed(var.name)
                        attributes = class_vars = None
                        break

                    if isinstance(var, Orange.data.Variable):
                        var = [var]

                    if i < len(self.data.domain.attributes):
                        attributes.extend(var)
                    else:
                        class_vars.extend(var)

                    progress.advance()

            if attributes is None:
                data = None
            else:
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
        methods = [self.get_method_for_column(i.row()) for i in indexes]

        def method_key(method):
            """
            Decompose method into its type and parameters.
            """
            # The return value should be hashable and  __eq__ comparable
            if isinstance(method, AsDefault):
                return AsDefault, (method.method,)
            elif isinstance(method, impute.Model):
                return impute.Model, (method.learner,)
            elif isinstance(method, impute.Default):
                return impute.Default, (method.default,)
            else:
                return type(method), None

        methods = set(method_key(m) for m in methods)
        selected_vars = [self.varmodel[index.row()] for index in indexes]
        has_discrete = any(var.is_discrete for var in selected_vars)
        fixed_value = None
        value_stack_enabled = False
        current_value_widget = None

        if len(methods) == 1:
            method_type, parameters = methods.pop()
            for i, m in enumerate(self.methods):
                if method_type == type(m):
                    self.variable_button_group.button(i).setChecked(True)

            if method_type is impute.Default:
                (fixed_value,) = parameters

        elif self.variable_button_group.checkedButton() is not None:
            # Uncheck the current button
            self.variable_button_group.setExclusive(False)
            self.variable_button_group.checkedButton().setChecked(False)
            self.variable_button_group.setExclusive(True)
            assert self.variable_button_group.checkedButton() is None

        for method, button in zip(self.methods,
                                  self.variable_button_group.buttons()):
            enabled = all(method.supports_variable(var) for var in
                          selected_vars)
            button.setEnabled(enabled)

        if not has_discrete:
            value_stack_enabled = True
            current_value_widget = self.value_double
        elif len(selected_vars) == 1:
            value_stack_enabled = True
            current_value_widget = self.value_combo
            self.value_combo.clear()
            self.value_combo.addItems(selected_vars[0].values)
        else:
            value_stack_enabled = False
            current_value_widget = None
            self.variable_button_group.button(self.AS_INPUT).setEnabled(False)

        self.value_stack.setEnabled(value_stack_enabled)
        if current_value_widget is not None:
            self.value_stack.setCurrentWidget(current_value_widget)
            if fixed_value is not None:
                if current_value_widget is self.value_combo:
                    self.value_combo.setCurrentIndex(fixed_value)
                elif current_value_widget is self.value_double:
                    self.value_double.setValue(fixed_value)
                else:
                    assert False

    def set_method_for_current_selection(self, method_index):
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, method_index)

    def set_method_for_indexes(self, indexes, method_index):
        if method_index == self.DEFAULT:
            for index in indexes:
                self.variable_methods.pop(index.row(), None)
        elif method_index == OWImpute.AS_INPUT:
            current = self.value_stack.currentWidget()
            if current is self.value_combo:
                value = self.value_combo.currentIndex()
            else:
                value = self.value_double.value()
            for index in indexes:
                method = impute.Default(default=value)
                self.variable_methods[index.row()] = method
        else:
            method = self.methods[method_index].copy()
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
        # The fixed 'Value' in the widget has been changed by the user.
        self.variable_button_group.button(self.AS_INPUT).setChecked(True)
        self.set_method_for_current_selection(self.AS_INPUT)

    def reset_variable_methods(self):
        indexes = list(map(self.varmodel.index, range(len(self.varmodel))))
        self.set_method_for_indexes(indexes, self.DEFAULT)
        self.variable_button_group.button(self.DEFAULT).setChecked(True)


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    argv = app.arguments()
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
