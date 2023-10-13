from typing import Optional

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.report.report import describe_data
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin


class TransformRunner:
    @staticmethod
    def run(
            data: Table,
            template_data: Table,
            state: TaskState
    ) -> Optional[Table]:
        if data is None or template_data is None:
            return None

        state.set_status("Transforming...")
        transformed_data = data.transform(template_data.domain)
        return transformed_data


class OWTransform(OWWidget, ConcurrentWidgetMixin):
    name = "Apply Domain"
    description = "Applies template domain on data table."
    category = "Transform"
    icon = "icons/Transform.svg"
    priority = 1230
    keywords = "apply domain, transform"

    class Inputs:
        data = Input("Data", Table, default=True)
        template_data = Input("Template Data", Table)

    class Outputs:
        transformed_data = Output("Transformed Data", Table)

    class Error(OWWidget.Error):
        error = Msg("An error occurred while transforming data.\n{}")

    resizing_enabled = False
    want_main_area = False
    buttons_area_orientation = None

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None  # type: Optional[Table]
        self.template_data = None  # type: Optional[Table]
        self.transformed_info = describe_data(None)  # type: OrderedDict

        box = gui.widgetBox(self.controlArea, True)
        gui.label(
            box, self, """
The widget takes Data, to which it re-applies transformations
that were applied to Template Data.

These include selecting a subset of variables as well as
computing variables from other variables appearing in the data,
like, for instance, discretization, feature construction, PCA etc.
""".strip(), box=True)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data = data

    @Inputs.template_data
    @check_sql_input
    def set_template_data(self, data):
        self.template_data = data

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        self.clear_messages()
        self.cancel()
        self.start(TransformRunner.run, self.data, self.template_data)

    def send_report(self):
        if self.data:
            self.report_data("Data", self.data)
        if self.template_data:
            self.report_domain("Template data", self.template_data.domain)
        if self.transformed_info:
            self.report_items("Transformed data", self.transformed_info)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Optional[Table]):
        self.transformed_info = describe_data(result)
        self.Outputs.transformed_data.send(result)

    def on_exception(self, ex):
        self.Error.error(ex)
        self.Outputs.transformed_data.send(None)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.preprocess import Discretize

    table = Table("iris")
    WidgetPreview(OWTransform).run(
        set_data=table, set_template_data=Discretize()(table))
