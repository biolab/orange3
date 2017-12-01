from Orange.data import Table, StringVariable
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output


class OWTranspose(OWWidget):
    name = "Transpose"
    description = "Transpose data table."
    icon = "icons/Transpose.svg"
    priority = 2000

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table, dynamic=False)

    GENERIC, FROM_META_ATTR = range(2)

    resizing_enabled = False
    want_main_area = False

    DEFAULT_PREFIX = "Feature"

    settingsHandler = DomainContextHandler()
    feature_type = ContextSetting(GENERIC)
    feature_name = ContextSetting("")
    feature_names_column = ContextSetting(None)
    auto_apply = Setting(True)

    class Error(OWWidget.Error):
        value_error = Msg("{}")

    def __init__(self):
        super().__init__()
        self.data = None

        box = gui.radioButtons(
            self.controlArea, self, "feature_type", box="Feature names",
            callback=lambda: self.apply())

        button = gui.appendRadioButton(box, "Generic")
        edit = gui.lineEdit(
            gui.indentedBox(box, gui.checkButtonOffsetHint(button)), self,
            "feature_name",
            placeholderText="Type a prefix ...", toolTip="Custom feature name")
        edit.editingFinished.connect(self._apply_editing)

        self.meta_button = gui.appendRadioButton(box, "From meta attribute:")
        self.feature_model = DomainModel(
            order=DomainModel.METAS, valid_types=StringVariable,
            alphabetical=True)
        self.feature_combo = gui.comboBox(
            gui.indentedBox(box, gui.checkButtonOffsetHint(button)), self,
            "feature_names_column", callback=self._feature_combo_changed,
            model=self.feature_model)

        self.apply_button = gui.auto_commit(
            self.controlArea, self, "auto_apply", "&Apply",
            box=False, commit=self.apply)
        self.apply_button.button.setAutoDefault(False)

        self.set_controls()

    def _apply_editing(self):
        self.feature_type = self.GENERIC
        self.feature_name = self.feature_name.strip()
        self.apply()

    def _feature_combo_changed(self):
        self.feature_type = self.FROM_META_ATTR
        self.apply()

    @Inputs.data
    def set_data(self, data):
        # Skip the context if the combo is empty: a context with
        # feature_model == None would then match all domains
        if self.feature_model:
            self.closeContext()
        self.data = data
        self.set_controls()
        if self.feature_model:
            self.openContext(data)
        self.apply()

    def set_controls(self):
        self.feature_model.set_domain(self.data and self.data.domain)
        self.meta_button.setEnabled(bool(self.feature_model))
        if self.feature_model:
            self.feature_names_column = self.feature_model[0]
            self.feature_type = self.FROM_META_ATTR
        else:
            self.feature_names_column = None

    def apply(self):
        self.clear_messages()
        transposed = None
        if self.data:
            try:
                transposed = Table.transpose(
                    self.data,
                    self.feature_type == self.FROM_META_ATTR and self.feature_names_column,
                    feature_name=self.feature_name or self.DEFAULT_PREFIX)
            except ValueError as e:
                self.Error.value_error(e)
        self.Outputs.data.send(transposed)

    def send_report(self):
        if self.feature_type == self.GENERIC:
            names = self.feature_name or self.DEFAULT_PREFIX
        else:
            names = "from meta attribute"
            if self.feature_names_column:
                names += "  '{}'".format(self.feature_names_column.name)
        self.report_items("", [("Feature names", names)])
        if self.data:
            self.report_data("Data", self.data)


if __name__ == "__main__":  # pragma: no cover
    from AnyQt.QtWidgets import QApplication

    app = QApplication([])
    ow = OWTranspose()
    d = Table("iris")
    ow.set_data(d)
    d = Table("zoo")
    ow.set_data(d)
    ow.show()
    app.exec_()
    ow.saveSettings()
