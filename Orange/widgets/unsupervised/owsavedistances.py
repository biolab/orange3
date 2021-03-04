from Orange.widgets.widget import Input, Msg
from Orange.misc import DistMatrix
from Orange.widgets.utils.save.owsavebase import OWSaveBase
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWSaveDistances(OWSaveBase):
    name = "Save Distance Matrix"
    description = "Save distance matrix to an output file."
    icon = "icons/SaveDistances.svg"
    keywords = ["distance matrix", "save"]

    filters = ["Distance File (*.dst)"]

    class Warning(OWSaveBase.Warning):
        table_not_saved = Msg("Associated data was not saved.")
        part_not_saved = Msg("Data associated with {} was not saved.")

    class Inputs:
        distances = Input("Distances", DistMatrix)

    @Inputs.distances
    def set_distances(self, data):
        self.data = data
        self.on_new_input()

    def do_save(self):
        dist = self.data
        dist.save(self.filename)
        skip_row = not dist.has_row_labels() and dist.row_items is not None
        skip_col = not dist.has_col_labels() and dist.col_items is not None
        self.Warning.table_not_saved(shown=skip_row and skip_col)
        self.Warning.part_not_saved("columns" if skip_col else "rows",
                                    shown=skip_row != skip_col,)

    def send_report(self):
        self.report_items((
            ("Input:", "none" if self.data is None else self._description()),
            ("File name", self.filename or "not set")))

    def _description(self):
        dist = self.data
        labels = " and ".join(
            filter(None, (dist.row_items is not None and "row",
                          dist.col_items is not None and "column")))
        if labels:
            labels = f"; {labels} labels"
        return f"{len(dist)}-dimensional matrix{labels}"


if __name__ == "__main__":
    from Orange.data import Table
    from Orange.distance import Euclidean
    WidgetPreview(OWSaveDistances).run(Euclidean(Table("iris")))
