from itertools import chain

from AnyQt.QtWidgets import QComboBox
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QDoubleValidator

from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from Orange.data import Variable, Table
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.classification import ColumnClassifier, ColumnLearner


class OWColumn(OWWidget):
    name = "Column as Model"
    description = "Predict values from columns."
    icon = "icons/ByColumn.svg"
    priority = 10
    keywords = "column"

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        learner = Output("Learner", ColumnLearner)
        model = Output("Model", ColumnClassifier)

    class Error(OWWidget.Error):
        no_class = Msg("Data has no class variable.")
        no_variables = Msg("No useful variables")
        invalid_probabilities = Msg(
            "Values must be between 0 and 1 (unless using logistic function).")

    column_hint: Variable = Setting(None, schema_only=True)
    offset = Setting(0)
    k = Setting(1)
    apply_logistic = Setting(0)
    auto_apply = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None
        self.column = None

        self.column_model = VariableListModel()
        box = gui.vBox(self.controlArea, True)
        hbox = gui.hBox(box)
        gui.label(hbox, self, "Predict values from")
        self.column_combo = combo = QComboBox()
        combo.setModel(self.column_model)
        combo.activated.connect(self.on_column_changed)
        box.layout().addWidget(combo)
        self.options = gui.vBox(box)
        self.bg = gui.radioButtons(
            self.options, self, "apply_logistic",
            ["Use values as probabilities"],
            tooltips=["For this, values must be betwwen 0 and 1."],
            callback=self.on_apply_logistic_changed)
        ibox = gui.hBox(self.options)
        ibox.layout().setSpacing(0)
        gui.appendRadioButton(
            self.bg, "Apply logistic function with", insertInto=ibox,
            tooltip="1 / [1 + exp(-k * (x - offset))]")
        gui.lineEdit(
            ibox,
            self, "offset", label="offset", orientation=Qt.Horizontal,
            valueType=float, validator=QDoubleValidator(),
            alignment=Qt.AlignRight, controlWidth=40,
            callback=self.on_log_pars_changed)
        gui.lineEdit(
            ibox,
            self, "k", label=", k", orientation=Qt.Horizontal,
            valueType=float, validator=QDoubleValidator(bottom=0),
            alignment=Qt.AlignRight, controlWidth=40,
            callback=self.on_log_pars_changed)
        gui.rubber(ibox)
        gui.auto_apply(self.controlArea, self, "auto_apply")
        self._update_controls()

    def on_column_changed(self, index):
        self.column = self.column_model[index]
        self.column_hint = self.column.name
        self._update_controls()
        self.commit.deferred()

    def on_apply_logistic_changed(self):
        self._update_controls()
        self.commit.deferred()

    def on_log_pars_changed(self):
        self.apply_logistic = 1
        self.commit.deferred()

    def _update_controls(self):
        self.options.setDisabled(self.column is None or self.column.is_discrete)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.data = None
        self.column_model.clear()
        self.column = None

        if data is not None:
            class_var = data.domain.class_var
            if class_var is None or not class_var.is_discrete:
                self.Error.no_class()
            else:
                classes = set(class_var.values)
                binary_class = len(classes) == 2
                self.column_model[:] = (
                    var
                    for var in chain(data.domain.attributes, data.domain.metas)
                    if ((var.is_continuous and binary_class) or
                        (var.is_discrete and
                         ColumnClassifier.check_value_sets(class_var, var))
                        ))
                if not self.column_model:
                    self.Error.no_variables()

        if not self.column_model:
            self.commit.now()
            return

        self.data = data
        if self.column_hint and \
                (var := self.data.domain[self.column_hint]) in self.column_model:
            self.column = var
            self.column_combo.setCurrentIndex(self.column_model.indexOf(self.column))
        else:
            self.column = self.column_model[0]
            self.column_combo.setCurrentIndex(0)
            self.column_hint = self.column.name

        self._update_controls()
        self.commit.now()

    @gui.deferred
    def commit(self):
        self.Error.invalid_probabilities.clear()
        if self.column is None:
            self.Outputs.learner.send(None)
            self.Outputs.model.send(None)
            return

        apply_logistic = self.column.is_continuous and self.apply_logistic
        learner = ColumnLearner(
            self.data.domain.class_var, self.column,
            self.offset if apply_logistic else None,
            self.k if apply_logistic else None)

        values = self.data.get_column(self.column)
        if not (apply_logistic or ColumnClassifier.check_prob_range(values)):
            self.Error.invalid_probabilities()
            model = None
        else:
            model = learner(self.data)

        self.Outputs.learner.send(learner)
        self.Outputs.model.send(model)

    def send_report(self):
        if self.column is None:
            return
        self.report_items((
            ("Predict values from", self.column),
            ("Apply logistic function", ["No", "Yes"][self.apply_logistic]),
            ("Offset", self.apply_logistic == 1 and self.offset),
            ("k", self.apply_logistic == 1 and self.k)
        ))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWColumn).run(Table("heart_disease"))
