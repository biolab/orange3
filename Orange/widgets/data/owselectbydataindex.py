import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.widgets import widget, gui
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details, \
    format_multiple_summaries
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.annotated_data import (create_annotated_table)


class OWSelectByDataIndex(widget.OWWidget):
    name = "Select by Data Index"
    description = "Match instances by index from data subset."
    icon = "icons/SelectByDataIndex.svg"
    priority = 1112

    class Inputs:
        data = Input("Data", Table)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        matching_data = Output("Matching Data", Table, replaces=["Data"], default=True)
        non_matching_data = Output("Unmatched Data", Table)
        # avoiding the default annotated output name (Data), as it was used
        # for Matching Data previously
        annotated_data = Output("Annotated Data", Table)

    want_main_area = False
    resizing_enabled = False

    class Warning(widget.OWWidget.Warning):
        instances_not_matching = widget.Msg("Input tables do not share any instances.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.data_subset = None

        self.model = itemmodels.VariableListModel()
        self.model_unique_with_id = itemmodels.VariableListModel()
        self.extra_model_unique = itemmodels.VariableListModel()
        self.extra_model_unique_with_id = itemmodels.VariableListModel()

        box = gui.hBox(self.controlArea, box=None)
        self.infoBoxData = gui.label(
            box, self, self.data_info_text(None), box="Data")
        self.infoBoxExtraData = gui.label(
            box, self, self.data_info_text(None), box="Data Subset")

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data = data
        self.infoBoxData.setText(self.data_info_text(data))

    @Inputs.data_subset
    @check_sql_input
    def set_data_subset(self, data):
        self.data_subset = data
        self.infoBoxExtraData.setText(self.data_info_text(data))

    def handleNewSignals(self):
        if self.data and self.data_subset:
            data_list = [('Data', self.data), ('Data subset', self.data_subset)]
            summary = f"{len(self.data)}, {len(self.data_subset)}"
            details = format_multiple_summaries(data_list)
            self.info.set_input_summary(summary, details, format=Qt.RichText)
        elif not self.data and not self.data_subset:
            self.info.set_input_summary(self.info.NoInput)
        else:
            summary = len(self.data) if self.data else len(self.data_subset)
            details = format_summary_details(self.data if self.data else self.data_subset)
            self.info.set_input_summary(summary, details)

        self._invalidate()

    @staticmethod
    def data_info_text(data):
        if data is None:
            return "No data."
        else:
            return "{}\n{} instances\n{} variables".format(
                data.name, len(data), len(data.domain) + len(data.domain.metas))

    def commit(self):
        self.Warning.instances_not_matching.clear()
        subset_ids = []
        if self.data_subset:
            subset_ids = self.data_subset.ids
        if not self.data:
            matching_output = None
            non_matching_output = None
            annotated_output = None
        else:
            if self.data_subset and \
                    not np.intersect1d(subset_ids, self.data.ids).size:
                self.Warning.instances_not_matching()
            row_sel = np.in1d(self.data.ids, subset_ids)
            matching_output = self.data[row_sel]
            non_matching_output = self.data[~row_sel]
            annotated_output = create_annotated_table(self.data, row_sel)

        summary = len(matching_output) if matching_output else self.info.NoOutput
        details = format_summary_details(matching_output) if matching_output else ""
        self.info.set_output_summary(summary, details)
        self.Outputs.matching_data.send(matching_output)
        self.Outputs.non_matching_data.send(non_matching_output)
        self.Outputs.annotated_data.send(annotated_output)

    def _invalidate(self):
        self.commit()

    def send_report(self):
        d_text = self.data_info_text(self.data).replace("\n", ", ")
        ds_text = self.data_info_text(self.data_subset).replace("\n", ", ")
        self.report_items("", [("Data", d_text), ("Data Subset", ds_text)])


if __name__ == "__main__":  # pragma: no cover
    iris = Table("iris.tab")
    WidgetPreview(OWSelectByDataIndex).run(
        set_data=iris,
        set_data_subset=iris[:20])
