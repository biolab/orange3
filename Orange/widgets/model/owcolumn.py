from itertools import chain

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QDoubleValidator

from orangewidget import gui

from Orange.classification.column import ColumnClassifier
from Orange.data import Table
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.utils.itemmodels import DomainModel, VariableListModel
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangewidget.gui import deferred
from orangewidget.settings import ContextSetting
from orangewidget.widget import Msg


class OWColumn(OWWidget):
    name = "By Column"
    description = "Predict values from columns."
    icon = "icons/ByColumn.svg"
    priority = 10
    keywords = "column"

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", ColumnClassifier)

    class Error(OWWidget.Error):
        no_class = Msg("Data has no class variable.")
        no_variables = Msg("No useful variables.")

    column = ContextSetting(None)
    k = ContextSetting(1)
    apply_logistic = ContextSetting(False)
    auto_apply = ContextSetting(True)

    def __init__(self):
        super().__init__()

        self.column_model = VariableListModel()
        gui.comboBox(
            self.controlArea, self, "column", box="Column",
            model=self.column_model,
            callback=self.on_column_changed)
        self.options = gui.vBox(self.controlArea, "Options")
        gui.checkBox(
            self.options, self, "apply_logistic", "Apply logistic function",
            callback=self.on_apply_logistic_changed)
        le = gui.lineEdit(
            gui.indentedBox(self.options),
            self, "k", label="k", orientation=Qt.Horizontal,
            valueType=float, validator=QDoubleValidator(bottom=0),
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.on_k_changed)
        gui.rubber(le.box)
        gui.auto_apply(self.controlArea, self, "auto_apply")
        self._update_controls()

    def on_column_changed(self):
        self._update_controls()
        self.commit.deferred()

    def on_apply_logistic_changed(self):
        self._update_controls()
        self.commit.deferred()

    def on_k_changed(self):
        self.commit.deferred()

    def _update_controls(self):
        self.options.setDisabled(self.column is None or self.column.is_discrete)
        self.controls.k.box.setDisabled(not self.apply_logistic)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.data = data
        if data is not None:
            class_var = data.domain.class_var
            if class_var is None or not class_var.is_discrete:
                self.Error.no_class()
                self.data = None
                self.column_model.clear()
            else:
                nvalues = len(class_var.values)
                self.column_model[:] = (
                    var
                    for var in chain(data.domain.attributes, data.domain.metas)
                    if var.is_continuous and nvalues == 2
                       or var.is_discrete and len(var.values) == nvalues)
                if not self.column_model:
                    self.Error.no_variables()
                    self.data = None
        if not self.column_model:
            self.column = None
            self.commit.now()
            return

        self.column = self.column_model[0]
        self._update_controls()
        self.commit.now()

    @deferred
    def commit(self):
        if self.column is None:
            model = None
        else:
            model = ColumnClassifier(
                self.data.domain.class_var, self.column,
                self.k if self.apply_logistic else None)
        self.Outputs.model.send(model)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWColumn).run(Table("heart_disease"))
