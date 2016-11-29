import os
import dill as pickle

from AnyQt.QtCore import QTimer
from AnyQt.QtWidgets import (
    QSizePolicy, QHBoxLayout, QComboBox, QStyle, QFileDialog
)
from Orange.base import Model

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import stdpaths

from Orange.widgets.classify import owsaveclassifier
from Orange.widgets.widget import Msg


class OWLoadClassifier(widget.OWWidget):
    name = "Load Classifier"
    description = "Load a classifier from an input file."
    priority = 3050
    icon = "icons/LoadClassifier.svg"

    outputs = [("Classifier", Model, widget.Dynamic)]

    #: List of recent filenames.
    history = Setting([])
    #: Current (last selected) filename or None.
    filename = Setting(None)

    class Error(widget.OWWidget.Error):
        load_error = Msg("An error occured while reading '{}'")

    FILTER = owsaveclassifier.OWSaveClassifier.FILTER

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.selectedIndex = -1

        box = gui.widgetBox(
            self.controlArea, self.tr("File"), orientation=QHBoxLayout()
        )

        self.filesCB = gui.comboBox(
            box, self, "selectedIndex", callback=self._on_recent)
        self.filesCB.setMinimumContentsLength(20)
        self.filesCB.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLength)

        self.loadbutton = gui.button(box, self, "...", callback=self.browse)
        self.loadbutton.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.loadbutton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        self.reloadbutton = gui.button(
            box, self, "Reload", callback=self.reload, default=True)
        self.reloadbutton.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload))
        self.reloadbutton.setSizePolicy(QSizePolicy.Maximum,
                                        QSizePolicy.Fixed)

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
            self.reloadbutton.setEnabled(False)

        if self.filename:
            QTimer.singleShot(0, lambda: self.load(self.filename))

    def browse(self):
        """Select a filename using an open file dialog."""
        if self.filename is None:
            startdir = stdpaths.Documents
        else:
            startdir = os.path.dirname(self.filename)

        filename, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open"), directory=startdir, filter=self.FILTER)

        if filename:
            self.load(filename)

    def reload(self):
        """Reload the current file."""
        self.load(self.filename)

    def load(self, filename):
        """Load the object from filename and send it to output."""
        try:
            with open(filename, "rb") as f:
                classifier = pickle.load(f)
        except (pickle.UnpicklingError, OSError, EOFError):
            self.Error.load_error(os.path.split(filename)[-1])
        else:
            self.Error.load_error.clear()
            self._remember(filename)
            self.send("Classifier", classifier)

    def _remember(self, filename):
        """
        Remember `filename` was accessed.
        """
        if filename in self.history:
            index = self.history.index(filename)
            del self.history[index]
            self.filesCB.removeItem(index)

        self.history.insert(0, filename)

        self.filesCB.insertItem(0, os.path.basename(filename),
                                userData=filename)
        self.selectedIndex = 0
        self.filename = filename
        self.reloadbutton.setEnabled(self.selectedIndex != -1)

    def _on_recent(self):
        self.load(self.history[self.selectedIndex])


def main():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])
    w = OWLoadClassifier()
    w.show()
    return app.exec_()

if __name__ == "__main__":
    import sys
    sys.exit(main())
