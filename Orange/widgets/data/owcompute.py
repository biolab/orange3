from typing import Optional

from Orange.data import Table
from Orange.data.dask import DaskTable
from Orange.widgets import gui
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin


class TransformRunner:
    @staticmethod
    def run(
            data: DaskTable,
            state: TaskState
    ) -> Optional[Table]:
        if data is None:
            return None
        d = data.compute()
        return d


class OWDaskCompute(OWWidget, ConcurrentWidgetMixin):
    name = "Dask Compute"
    description = "Computes Dask Table into memory."
    category = "Data"
    icon = "icons/Unknown.svg"
    priority = 9999
    keywords = "dask, compute"

    class Inputs:
        data = Input("Data", DaskTable, default=True)

    class Outputs:
        transformed_data = Output("Computed Data", Table)

    class Error(OWWidget.Error):
        error = Msg("An error occurred.\n{}")

    resizing_enabled = False
    want_main_area = False
    buttons_area_orientation = None

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None  # type: Optional[Table]

        box = gui.widgetBox(self.controlArea, True)
        gui.label(
            box, self, """
The widget takes Data and computes it so that is is saved to memory.
""".strip(), box=True)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data = data

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        self.clear_messages()
        self.cancel()
        self.start(TransformRunner.run, self.data)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Optional[Table]):
        self.Outputs.transformed_data.send(result)

    def on_exception(self, ex):
        self.Error.error(ex)
        self.Outputs.transformed_data.send(None)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.tests.test_dasktable import temp_dasktable
    table = temp_dasktable("iris")
    WidgetPreview(OWDaskCompute).run(set_data=table)
