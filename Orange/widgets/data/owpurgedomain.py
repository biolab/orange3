from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.preprocess.remove import Remove
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input


class OWPurgeDomain(widget.OWWidget):
    name = "Purge Domain"
    description = "Remove redundant values and features from the data set. " \
                  "Sort values."
    icon = "icons/PurgeDomain.svg"
    category = "Data"
    keywords = ["data", "purge", "domain"]

    inputs = [("Data", Table, "setData")]
    outputs = [("Data", Table)]

    removeValues = Setting(1)
    removeAttributes = Setting(1)
    removeClasses = Setting(1)
    removeClassAttribute = Setting(1)
    removeMetaAttributeValues = Setting(1)
    removeMetaAttributes = Setting(1)
    autoSend = Setting(False)
    sortValues = Setting(True)
    sortClasses = Setting(True)

    want_main_area = False
    resizing_enabled = False
    buttons_area_orientation = Qt.Vertical

    feature_options = (('sortValues', 'Sort discrete feature values'),
                       ('removeValues', 'Remove unused feature values'),
                       ('removeAttributes', 'Remove constant features'))

    class_options = (('sortClasses', 'Sort discrete class variable values'),
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

        boxAt = gui.vBox(self.controlArea, "Features")
        for not_first, (value, label) in enumerate(self.feature_options):
            if not_first:
                gui.separator(boxAt, 2)
            gui.checkBox(boxAt, self, value, label,
                         callback=self.optionsChanged)

        boxAt = gui.vBox(self.controlArea, "Classes", addSpace=True)
        for not_first, (value, label) in enumerate(self.class_options):
            if not_first:
                gui.separator(boxAt, 2)
            gui.checkBox(boxAt, self, value, label,
                         callback=self.optionsChanged)

        boxAt = gui.vBox(self.controlArea, "Meta attributes", addSpace=True)
        for not_first, (value, label) in enumerate(self.meta_options):
            if not_first:
                gui.separator(boxAt, 2)
            gui.checkBox(boxAt, self, value, label,
                         callback=self.optionsChanged)

        box3 = gui.vBox(self.controlArea, 'Statistics', addSpace=True)
        for i, (label, value) in enumerate(self.stat_labels):
            # add a separator after each group of three
            if i != 0 and i % 3 == 0:
                gui.separator(box3, 2)
            gui.label(box3, self, "{}: %({})s".format(label, value))

        gui.auto_commit(self.buttonsArea, self, "autoSend", "Apply",
                        orientation=Qt.Horizontal)
        gui.rubber(self.controlArea)

    @check_sql_input
    def setData(self, dataset):
        if dataset is not None:
            self.data = dataset
            self.unconditional_commit()
        else:
            self.removedAttrs = "-"
            self.reducedAttrs = "-"
            self.resortedAttrs = "-"
            self.removedClasses = "-"
            self.reducedClasses = "-"
            self.resortedClasses = "-"
            self.removedMetas = "-"
            self.reducedMetas = "-"
            self.send("Data", None)
            self.data = None

    def optionsChanged(self):
        self.commit()

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
        data = remover(self.data)
        attr_res, class_res, meta_res = remover.attr_results, remover.class_results, remover.meta_results

        self.removedAttrs = attr_res['removed']
        self.reducedAttrs = attr_res['reduced']
        self.resortedAttrs = attr_res['sorted']

        self.removedClasses = class_res['removed']
        self.reducedClasses = class_res['reduced']
        self.resortedClasses = class_res['sorted']

        self.removedMetas = meta_res['removed']
        self.reducedMetas = meta_res['reduced']

        self.send("Data", data)

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


if __name__ == "__main__":
    appl = QtGui.QApplication([])
    ow = OWPurgeDomain()
    data = Table("car.tab")
    subset = [inst for inst in data
              if inst["buying"] == "v-high"]
    subset = Table(data.domain, subset)
    # The "buying" should be removed and the class "y" reduced
    ow.setData(subset)
    ow.show()
    appl.exec_()
    ow.saveSettings()
