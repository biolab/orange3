from AnyQt.QtWidgets import QApplication

import numpy as np

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Input, Output


class OWSelectSubset(widget.OWWidget):
    name = "Select Subset"
    description = "Select same instances"
    icon = "icons/Unknown.svg"
    priority = 9999

    class Inputs:
        data = Input("Data", Orange.data.Table)
        data_subset = Input("Data Subset", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    matching = settings.Setting(0)

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
            box, self, self.dataInfoText(None), box="Data")
        self.infoBoxExtraData = gui.label(
            box, self, self.dataInfoText(None), box="Data Subset")

        grp = gui.radioButtonsInBox(
            self.controlArea, self, "matching", box="Output",
            callback=self._invalidate)
        self.attr_boxes = []

        gui.appendRadioButton(grp, "Matching data")
        gui.appendRadioButton(grp, "Other data")

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        self.data = data
        self.infoBoxData.setText(self.dataInfoText(data))

    @Inputs.data_subset
    @check_sql_input
    def setDataSubset(self, data):
        self.data_subset = data
        self.infoBoxExtraData.setText(self.dataInfoText(data))

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
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
        if self.data is None or len(self.data) == 0:
            output_data = None
        else:
            if len(np.intersect1d(subset_ids, self.data.ids)) == 0:
                self.Warning.instances_not_matching()
            subset_indices = np.in1d(self.data.ids, subset_ids)
            if self.matching == 1:
                subset_indices = ~subset_indices
            output_data = self.data[subset_indices]
        self.Outputs.data.send(output_data)

    def _invalidate(self):
        self.commit()

    def send_report(self):
        matching = ("Matching data", "Other data")
        self.report_items((
            ("Output", matching[self.matching]),
            ))


def main():
    app = QApplication([])
    w = OWSelectSubset()
    data = Orange.data.Table("iris.tab")
    data_subset = data[:20]
    w.setData(data)
    w.setDataSubset(data_subset)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
