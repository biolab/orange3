from AnyQt.QtWidgets import QFrame

from Orange.data import Table
from Orange.preprocess.remove import Remove
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


class OWPurgeDomain(widget.OWWidget):
    name = "Purge Domain"
    description = "Remove redundant values and features from the dataset. " \
                  "Sort values."
    icon = "icons/PurgeDomain.svg"
    category = "Data"
    keywords = ["remove", "delete", "unused"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    removeValues = Setting(1)
    removeAttributes = Setting(1)
    removeClasses = Setting(1)
    removeClassAttribute = Setting(1)
    removeMetaAttributeValues = Setting(1)
    removeMetaAttributes = Setting(1)
    autoSend = Setting(True)
    sortValues = Setting(True)
    sortClasses = Setting(True)

    want_main_area = False
    resizing_enabled = False

    feature_options = (('sortValues', 'Sort categorical feature values'),
                       ('removeValues', 'Remove unused feature values'),
                       ('removeAttributes', 'Remove constant features'))

    class_options = (('sortClasses', 'Sort categorical class values'),
                     ('removeClasses', 'Remove unused class variable values'),
                     ('removeClassAttribute', 'Remove constant class variables'))

    meta_options = (('removeMetaAttributeValues', 'Remove unused meta attribute values'),
                    ('removeMetaAttributes', 'Remove constant meta attributes'))

    stat_labels = (('Sorted features', 'resortedAttrs'),
                   ('Reduced features', 'reducedAttrs'),
                   ('Removed features', 'removedAttrs'),
                   ('Sorted classes', 'resortedClasses'),
                   ('Reduced classes', 'reducedClasses'),
                   ('Removed classes', 'removedClasses'),
                   ('Reduced metas', 'reducedMetas'),
                   ('Removed metas', 'removedMetas'))

    def __init__(self):
        super().__init__()
        self.data = None

        self.removedAttrs = "-"
        self.reducedAttrs = "-"
        self.resortedAttrs = "-"
        self.removedClasses = "-"
        self.reducedClasses = "-"
        self.resortedClasses = "-"
        self.removedMetas = "-"
        self.reducedMetas = "-"

        def add_line(parent):
            frame = QFrame()
            frame.setFrameShape(QFrame.HLine)
            frame.setFrameShadow(QFrame.Sunken)
            parent.layout().addWidget(frame)

        boxAt = gui.vBox(self.controlArea, "Features")
        for value, label in self.feature_options:
            gui.checkBox(boxAt, self, value, label,
                         callback=self.commit.deferred)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "Sorted: %(resortedAttrs)s, "
                  "reduced: %(reducedAttrs)s, removed: %(removedAttrs)s")

        boxAt = gui.vBox(self.controlArea, "Classes")
        for value, label in self.class_options:
            gui.checkBox(boxAt, self, value, label,
                         callback=self.commit.deferred)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "Sorted: %(resortedClasses)s,"
                  "reduced: %(reducedClasses)s, removed: %(removedClasses)s")

        boxAt = gui.vBox(self.controlArea, "Meta attributes")
        for value, label in self.meta_options:
            gui.checkBox(boxAt, self, value, label,
                         callback=self.commit.deferred)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "Reduced: %(reducedMetas)s, removed: %(removedMetas)s")

        gui.auto_send(self.buttonsArea, self, "autoSend")

    @Inputs.data
    @check_sql_input
    def setData(self, dataset):
        if dataset is not None:
            self.data = dataset
            self.commit.now()
        else:
            self.removedAttrs = "-"
            self.reducedAttrs = "-"
            self.resortedAttrs = "-"
            self.removedClasses = "-"
            self.reducedClasses = "-"
            self.resortedClasses = "-"
            self.removedMetas = "-"
            self.reducedMetas = "-"
            self.Outputs.data.send(None)
            self.data = None

    @gui.deferred
    def commit(self):
        if self.data is None:
            return

        attr_flags = sum([Remove.SortValues * self.sortValues,
                          Remove.RemoveConstant * self.removeAttributes,
                          Remove.RemoveUnusedValues * self.removeValues])
        class_flags = sum([Remove.SortValues * self.sortClasses,
                           Remove.RemoveConstant * self.removeClassAttribute,
                           Remove.RemoveUnusedValues * self.removeClasses])
        meta_flags = sum([Remove.RemoveConstant * self.removeMetaAttributes,
                          Remove.RemoveUnusedValues * self.removeMetaAttributeValues])
        remover = Remove(attr_flags, class_flags, meta_flags)
        cleaned = remover(self.data)
        attr_res, class_res, meta_res = \
            remover.attr_results, remover.class_results, remover.meta_results

        self.removedAttrs = attr_res['removed']
        self.reducedAttrs = attr_res['reduced']
        self.resortedAttrs = attr_res['sorted']

        self.removedClasses = class_res['removed']
        self.reducedClasses = class_res['reduced']
        self.resortedClasses = class_res['sorted']

        self.removedMetas = meta_res['removed']
        self.reducedMetas = meta_res['reduced']

        self.Outputs.data.send(cleaned)

    def send_report(self):
        def list_opts(opts):
            return "; ".join(label.lower()
                             for value, label in opts
                             if getattr(self, value)) or "no changes"

        self.report_items("Settings", (
            ("Features", list_opts(self.feature_options)),
            ("Classes", list_opts(self.class_options)),
            ("Metas", list_opts(self.meta_options))))
        if self.data:
            self.report_items("Statistics", (
                (label, getattr(self, value))
                for label, value in self.stat_labels
            ))


if __name__ == "__main__":  # pragma: no cover
    data = Table.from_url("https://datasets.biolab.si/core/car.tab")
    subset = [inst for inst in data if inst["buying"] == "v-high"]
    subset = Table(data.domain, subset)
    # The "buying" should be removed and the class "y" reduced
    WidgetPreview(OWPurgeDomain).run(subset)
