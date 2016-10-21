import os.path

from AnyQt.QtWidgets import QFileDialog

from Orange.misc import DistMatrix
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting


class OWSaveDistances(widget.OWWidget):
    name = "Save Distance Matrix"
    description = "Save distance matrix to an output file."
    icon = "icons/SaveDistances.svg"
    category = "Unsupervised"
    keywords = ["distance matrix", "save"]

    inputs = [("Distances", DistMatrix, "set_distances")]

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")
    auto_save = Setting(False)

    def __init__(self):
        super().__init__()
        self.distances = None
        self.filename = ""

        self.save = gui.auto_commit(
            self.controlArea, self, "auto_save", "Save", box=False,
            commit=self.save_file, callback=self.adjust_label,
            disabled=True, addSpace=True)
        self.saveAs = gui.button(
            self.controlArea, self, "Save As...",
            callback=self.save_file_as, disabled=True)
        self.saveAs.setMinimumWidth(300)
        self.adjustSize()

    def adjust_label(self):
        if self.filename:
            filename = os.path.split(self.filename)[1]
            text = ["Save as '{}'", "Auto save as '{}'"][self.auto_save]
            self.save.button.setText(text.format(filename))

    def set_distances(self, distances):
        self.distances = distances
        self.save.setDisabled(distances is None)
        self.saveAs.setDisabled(distances is None)
        if distances is not None:
            self.save_file()

    def save_file_as(self):
        file_name = self.filename or self.last_dir or os.path.expanduser("~")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Select file", file_name, 'Distance files (*.dst)')
        if not filename:
            return
        self.filename = filename
        self.unconditional_save_file()
        self.last_dir = os.path.split(self.filename)[0]
        self.adjust_label()

    def save_file(self):
        dist = self.distances
        if dist is None:
            return
        if not self.filename:
            self.save_file_as()
        else:
            dist.save(self.filename)
            skip_row = not dist.has_row_labels() and dist.row_items is not None
            skip_col = not dist.has_col_labels() and dist.col_items is not None
            if skip_row and skip_col:
                self.warning("Associated data table was not saved")
            elif skip_row or skip_col:
                self.warning("Data associated with {} was not saved".
                             format(["rows", "columns"][skip_col]))
            else:
                self.warning()


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    table = Table("iris")
    ow = OWSaveDistances()
    ow.show()
    ow.set_distances(table)
    a.exec()
    ow.saveSettings()
