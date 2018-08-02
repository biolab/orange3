import numpy as np

from AnyQt.QtWidgets import QApplication

from Orange.data import Table
from Orange.widgets import widget, gui
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Input, Output


class OWSelectByDataIndex(widget.OWWidget):
    name = "Select by Data Index"
    description = "Match instances by index from data subset."
    icon = "icons/SelectByDataIndex.svg"
    priority = 1112

    class Inputs:
        data = Input("Data", Table)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        data = Output("Data", Table)

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
        self._invalidate()

    def data_info_text(self, data):
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
            output_data = None
        else:
            if self.data_subset and len(np.intersect1d(subset_ids, self.data.ids)) == 0:
                self.Warning.instances_not_matching()
            subset_indices = np.in1d(self.data.ids, subset_ids)
            output_data = self.data[subset_indices]
        self.Outputs.data.send(output_data)

    def _invalidate(self):
        self.commit()


def main():
    app = QApplication([])
    w = OWSelectByDataIndex()
    data = Table("iris.tab")
    data_subset = data[:20]
    w.set_data(data)
    w.set_data_subset(data_subset)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
