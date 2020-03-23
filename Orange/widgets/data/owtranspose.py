from Orange.data import Table, ContinuousVariable, StringVariable
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output


class OWTranspose(OWWidget):
    name = "Transpose"
    description = "Transpose data table."
    icon = "icons/Transpose.svg"
    priority = 2000
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table, dynamic=False)

    GENERIC, FROM_VAR = range(2)

    resizing_enabled = False
    want_main_area = False

    DEFAULT_PREFIX = "Feature"

    settingsHandler = DomainContextHandler()
    feature_type = ContextSetting(GENERIC)
    feature_name = ContextSetting("")
    feature_names_column = ContextSetting(None)
    auto_apply = Setting(True)

    class Warning(OWWidget.Warning):
        duplicate_names = Msg("Values are not unique.\nTo avoid multiple "
                              "features with the same name, values \nof "
                              "'{}' have been augmented with indices.")
        discrete_attrs = Msg("Categorical features have been encoded as numbers.")

    class Error(OWWidget.Error):
        value_error = Msg("{}")

    def __init__(self):
        super().__init__()
        self.data = None

        # self.apply is changed later, pylint: disable=unnecessary-lambda
        box = gui.radioButtons(
            self.controlArea, self, "feature_type", box="Feature names",
            callback=lambda: self.apply())

        button = gui.appendRadioButton(box, "Generic")
        edit = gui.lineEdit(
            gui.indentedBox(box, gui.checkButtonOffsetHint(button)), self,
            "feature_name",
            placeholderText="Type a prefix ...", toolTip="Custom feature name")
        edit.editingFinished.connect(self._apply_editing)

        self.meta_button = gui.appendRadioButton(box, "From variable:")
        self.feature_model = DomainModel(
            valid_types=(ContinuousVariable, StringVariable),
            alphabetical=False)
        self.feature_combo = gui.comboBox(
            gui.indentedBox(box, gui.checkButtonOffsetHint(button)), self,
            "feature_names_column", contentsLength=12, searchable=True,
            callback=self._feature_combo_changed, model=self.feature_model)

        self.apply_button = gui.auto_apply(self.controlArea, self, box=False, commit=self.apply)
        self.apply_button.button.setAutoDefault(False)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

        self.set_controls()

    def _apply_editing(self):
        self.feature_type = self.GENERIC
        self.feature_name = self.feature_name.strip()
        self.apply()

    def _feature_combo_changed(self):
        self.feature_type = self.FROM_VAR
        self.apply()

    @Inputs.data
    def set_data(self, data):
        # Skip the context if the combo is empty: a context with
        # feature_model == None would then match all domains
        if self.feature_model:
            self.closeContext()
        self.data = data
        if data:
            self.info.set_input_summary(len(data), format_summary_details(data))
        else:
            self.info.set_input_summary(self.info.NoInput)
        self.set_controls()
        if self.feature_model:
            self.openContext(data)
        self.unconditional_apply()

    def set_controls(self):
        self.feature_model.set_domain(self.data and self.data.domain)
        self.meta_button.setEnabled(bool(self.feature_model))
        if self.feature_model:
            self.feature_names_column = self.feature_model[0]
            self.feature_type = self.FROM_VAR
        else:
            self.feature_names_column = None

    def apply(self):
        self.clear_messages()
        transposed = None
        if self.data:
            try:
                variable = self.feature_type == self.FROM_VAR and \
                           self.feature_names_column
                transposed = Table.transpose(
                    self.data, variable,
                    feature_name=self.feature_name or self.DEFAULT_PREFIX)
                if variable:
                    names = self.data.get_column_view(variable)[0]
                    if len(names) != len(set(names)):
                        self.Warning.duplicate_names(variable)
                if self.data.domain.has_discrete_attributes():
                    self.Warning.discrete_attrs()
                self.info.set_output_summary(len(transposed),
                                             format_summary_details(transposed))
            except ValueError as e:
                self.Error.value_error(e)
        else:
            self.info.set_output_summary(self.info.NoOutput)
        self.Outputs.data.send(transposed)

    def send_report(self):
        if self.feature_type == self.GENERIC:
            names = self.feature_name or self.DEFAULT_PREFIX
        else:
            names = "from variable"
            if self.feature_names_column:
                names += "  '{}'".format(self.feature_names_column.name)
        self.report_items("", [("Feature names", names)])
        if self.data:
            self.report_data("Data", self.data)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWTranspose).run(Table("iris"))
