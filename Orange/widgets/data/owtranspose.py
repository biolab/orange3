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

    resizing_enabled = False
    want_main_area = False

    settingsHandler = DomainContextHandler(metas_in_res=True)
    feature_type = ContextSetting(0)
    feature_names_column = ContextSetting(None)
    auto_apply = Setting(True)

    class Error(OWWidget.Error):
        value_error = Msg("{}")

    def __init__(self):
        super().__init__()
        self.data = None

        # GUI
        box = gui.vBox(self.controlArea, "Feature names")
        self.feature_radio = gui.radioButtonsInBox(
            box, self, "feature_type", callback=lambda: self.apply(),
            btnLabels=["Generic", "From meta attribute:"])

        self.feature_model = DomainModel(
            order=DomainModel.METAS, valid_types=StringVariable,
            alphabetical=True)
        self.feature_combo = gui.comboBox(
            gui.indentedBox(
                box, gui.checkButtonOffsetHint(self.feature_radio.buttons[0])),
            self, "feature_names_column", callback=self._feature_combo_changed,
            model=self.feature_model)

        self.apply_button = gui.auto_commit(
            self.controlArea, self, "auto_apply", "&Apply",
            box=False, commit=self.apply)

    def _feature_combo_changed(self):
        self.feature_type = 1
        self.apply()

    @Inputs.data
    def set_data(self, data):
        # Skip the context if the combo is empty: a context with
        # feature_model == None would then match all domains
        if self.feature_model:
            self.closeContext()
        self.data = data
        self.update_controls()
        if self.data is not None and self.feature_model:
            self.openContext(data)
        self.apply()

    def update_controls(self):
        self.feature_model.set_domain(None)
        if self.data:
            self.feature_model.set_domain(self.data.domain)
            if self.feature_model:
                self.feature_names_column = self.feature_model[0]
        enabled = bool(self.feature_model)
        self.feature_radio.buttons[1].setEnabled(enabled)
        self.feature_combo.setEnabled(enabled)
        self.feature_type = int(enabled)

    def apply(self):
        self.clear_messages()
        transposed = None
        if self.data:
            try:
                transposed = Table.transpose(
                    self.data, self.feature_type and self.feature_names_column)
            except ValueError as e:
                self.Error.value_error(e)
        self.Outputs.data.send(transposed)

    def send_report(self):
        text = "from meta attribute: {}".format(self.feature_names_column) \
            if self.feature_type else "generic"
        self.report_items("", [("Feature names", text)])
        if self.data:
            self.report_data("Data", self.data)


if __name__ == "__main__":
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
