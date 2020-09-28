import os
import pickle
from typing import Any, Dict

from AnyQt.QtWidgets import QSizePolicy, QStyle, QFileDialog
from AnyQt.QtCore import QTimer

from orangewidget.workflow.drophandler import SingleFileDropHandler

from Orange.base import Model
from Orange.widgets import widget, gui
from Orange.widgets.model import owsavemodel
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin, RecentPath
from Orange.widgets.utils import stdpaths
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Output


class OWLoadModel(widget.OWWidget, RecentPathsWComboMixin):
    name = "Load Model"
    description = "Load a model from an input file."
    priority = 3050
    replaces = ["Orange.widgets.classify.owloadclassifier.OWLoadClassifier"]
    icon = "icons/LoadModel.svg"
    keywords = ["file", "open", "model"]

    class Outputs:
        model = Output("Model", Model)

    class Error(widget.OWWidget.Error):
        load_error = Msg("An error occured while reading '{}'")

    FILTER = ";;".join(owsavemodel.OWSaveModel.filters)

    want_main_area = False
    buttons_area_orientation = None
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)
        self.loaded_file = ""

        vbox = gui.vBox(self.controlArea, "File")
        box = gui.hBox(vbox)
        self.file_combo.setMinimumWidth(300)
        box.layout().addWidget(self.file_combo)
        self.file_combo.activated[int].connect(self.select_file)

        button = gui.button(box, self, '...', callback=self.browse_file)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Fixed)

        button = gui.button(
            box, self, "Reload", callback=self.reload, default=True)
        button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.set_file_list()
        QTimer.singleShot(0, self.open_file)

    def browse_file(self):
        start_file = self.last_path() or stdpaths.Documents
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Distance File', start_file, self.FILTER)
        if not filename:
            return
        self.add_path(filename)
        self.open_file()

    def select_file(self, n):
        super().select_file(n)
        self.open_file()

    def reload(self):
        self.open_file()

    def open_file(self):
        self.clear_messages()
        fn = self.last_path()
        if not fn:
            return
        try:
            with open(fn, "rb") as f:
                model = pickle.load(f)
        except (pickle.UnpicklingError, OSError, EOFError):
            self.Error.load_error(os.path.split(fn)[-1])
            self.Outputs.model.send(None)
        else:
            self.Outputs.model.send(model)


class OWLoadModelDropHandler(SingleFileDropHandler):
    WIDGET = OWLoadModel

    def canDropFile(self, path: str) -> bool:
        return path.endswith(".pkcls")

    def parametersFromFile(self, path: str) -> Dict[str, Any]:
        r = RecentPath(os.path.abspath(path), None, None,
                       os.path.basename(path))
        parameters = {"recent_paths": [r]}
        return parameters


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWLoadModel).run()
