import os

import dill as pickle
from AnyQt.QtWidgets import (
    QComboBox, QStyle, QSizePolicy, QFileDialog, QApplication
)

from Orange.base import Model
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import stdpaths


class OWSaveModel(widget.OWWidget):
    name = "Save Model"
    description = "Save a trained model to an output file."
    icon = "icons/SaveModel.svg"
    replaces = ["Orange.widgets.classify.owsaveclassifier.OWSaveClassifier"]
    priority = 3000

    inputs = [("Model", Model, "setModel")]

    #: Current (last selected) filename or None.
    filename = Setting(None)
    #: A list of recent filenames.
    history = Setting([])

    FILE_EXT = '.pkcls'
    FILTER = "Pickled model (*" + FILE_EXT + ");;All Files (*)"

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.selectedIndex = -1
        #: input model
        self.model = None

        box = gui.hBox(self.controlArea, self.tr("File"))
        self.filesCB = gui.comboBox(box, self, "selectedIndex",
                                    callback=self._on_recent)
        self.filesCB.setMinimumContentsLength(20)
        self.filesCB.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLength)

        button = gui.button(
            box, self, "...", callback=self.browse, default=True
        )
        button.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon)
        )
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        self.savebutton = gui.button(
            self.controlArea, self, "Save", callback=self.savecurrent,
            default=True
        )
        self.savebutton.setEnabled(False)

        # filter valid existing filenames
        self.history = list(filter(os.path.isfile, self.history))[:20]
        for filename in self.history:
            self.filesCB.addItem(os.path.basename(filename), userData=filename)

        # restore the current selection if the filename is
        # in the history list
        if self.filename in self.history:
            self.selectedIndex = self.history.index(self.filename)
        else:
            self.selectedIndex = -1
            self.filename = None

    def setModel(self, model):
        """Set input model."""
        self.model = model
        self.savebutton.setEnabled(
            not (model is None or self.filename is None))

    def save(self, filename):
        """Save the model to filename (model must not be None)."""
        assert self.model is not None
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.model, f)
        except pickle.PicklingError:
            raise
        except os.error:
            raise
        else:
            self._remember(filename)

    def savecurrent(self):
        """
        Save the model to current selected filename.

        Do nothing if model or current filename are None.

        """
        if self.model is not None and self.filename is not None:
            self.save(self.filename)

    def browse(self):
        """Select a filename using a Save file dialog."""
        if self.filename is None:
            startdir = stdpaths.Documents
        else:
            startdir = os.path.dirname(self.filename)

        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save"), directory=startdir, filter=self.FILTER
        )
        if filename:
            if not filename.endswith(self.FILE_EXT):
                filename += self.FILE_EXT
            if self.model is not None:
                self.save(filename)
            else:
                self._remember(filename)

    def _on_recent(self):
        filename = self.history[self.selectedIndex]
        self._remember(filename)

    def _remember(self, filename):
        if filename in self.history:
            index = self.history.index(filename)
            del self.history[index]
            self.filesCB.removeItem(index)

        self.history.insert(0, filename)
        self.filesCB.insertItem(0, os.path.basename(filename),
                                userData=filename)

        self.filename = filename
        self.selectedIndex = 0
        self.savebutton.setEnabled(
            not (self.model is None or self.filename is None))


def main():
    app = QApplication([])
    w = OWSaveModel()
    w.show()
    return app.exec_()

if __name__ == "__main__":
    import sys
    sys.exit(main())

