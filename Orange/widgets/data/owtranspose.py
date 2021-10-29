from typing import Optional, Union

from Orange.data import Table, ContinuousVariable, StringVariable, Variable
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output


def run(data: Table,
        variable: Optional[Union[Variable, bool]],
        feature_name: str,
        remove_redundant_inst: bool,
        state: TaskState
        ) -> Table:
    if not data:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    return Table.transpose(data, variable, feature_name=feature_name,
                           remove_redundant_inst=remove_redundant_inst,
                           progress_callback=callback)


class OWTranspose(OWWidget, ConcurrentWidgetMixin):
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
    remove_redundant_inst = ContextSetting(False)
    auto_apply = Setting(True)

    class Warning(OWWidget.Warning):
        duplicate_names = Msg("Values are not unique.\nTo avoid multiple "
                              "features with the same name, values \nof "
                              "'{}' have been augmented with indices.")
        discrete_attrs = Msg("Categorical features have been encoded as numbers.")

    class Error(OWWidget.Error):
        value_error = Msg("{}")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None

        # self.apply is changed later, pylint: disable=unnecessary-lambda
        box = gui.radioButtons(
            self.controlArea, self, "feature_type", box="Feature names",
            callback=self.commit.deferred)

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

        self.remove_check = gui.checkBox(
            gui.indentedBox(box, gui.checkButtonOffsetHint(button)), self,
            "remove_redundant_inst", "Remove redundant instance",
            callback=self.commit.deferred)

        gui.auto_apply(self.buttonsArea, self)

        self.set_controls()

    def _apply_editing(self):
        self.feature_type = self.GENERIC
        self.feature_name = self.feature_name.strip()
        self.commit.deferred()

    def _feature_combo_changed(self):
        self.feature_type = self.FROM_VAR
        self.commit.deferred()

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
        self.commit.now()

    def set_controls(self):
        self.feature_model.set_domain(self.data.domain if self.data else None)
        self.meta_button.setEnabled(bool(self.feature_model))
        if self.feature_model:
            self.feature_names_column = self.feature_model[0]
            self.feature_type = self.FROM_VAR
        else:
            self.feature_names_column = None

    @gui.deferred
    def commit(self):
        self.clear_messages()
        variable = self.feature_type == self.FROM_VAR and \
            self.feature_names_column
        if variable and self.data:
            names = self.data.get_column_view(variable)[0]
            if len(names) != len(set(names)):
                self.Warning.duplicate_names(variable)
        if self.data and self.data.domain.has_discrete_attributes():
            self.Warning.discrete_attrs()
        feature_name = self.feature_name or self.DEFAULT_PREFIX
        self.start(run, self.data, variable,
                   feature_name, self.remove_redundant_inst)

    def on_partial_result(self, _):
        pass

    def on_done(self, transposed: Optional[Table]):
        self.Outputs.data.send(transposed)

    def on_exception(self, ex: Exception):
        if isinstance(ex, ValueError):
            self.Error.value_error(ex)
        else:
            raise ex

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

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
