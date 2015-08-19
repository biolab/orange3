from PyQt4 import QtGui

import Orange

from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.preprocess.remove import Remove


class OWPurgeDomain(widget.OWWidget):
    name = "Purge Domain"
    description = "Remove redundant values and features from the data set. " \
                  "Sorts values."
    icon = "icons/PurgeDomain.svg"
    category = "Data"
    keywords = ["data", "purge", "domain"]

    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Data", Orange.data.Table)]

    removeValues = Setting(1)
    removeAttributes = Setting(1)
    removeClassAttribute = Setting(1)
    removeClasses = Setting(1)
    autoSend = Setting(False)
    sortValues = Setting(True)
    sortClasses = Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.preRemoveValues = 1
        self.preRemoveClasses = 1

        self.removedAttrs = "-"
        self.reducedAttrs = "-"
        self.resortedAttrs = "-"
        self.removedClasses = "-"
        self.reducedClasses = "-"
        self.resortedClasses = "-"

        boxAt = gui.widgetBox(self.controlArea, "Attributes")
        gui.checkBox(boxAt, self, 'sortValues', 'Sort attribute values',
                     callback=self.optionsChanged)
        gui.separator(boxAt, 2)
        rua = gui.checkBox(
            boxAt, self, "removeAttributes",
            "Remove attributes with less than two values",
            callback=self.removeAttributesChanged)
        ruv = gui.checkBox(
            gui.indentedBox(boxAt, sep=gui.checkButtonOffsetHint(rua)),
            self,
            "removeValues",
            "Remove unused attribute values",
            callback=self.optionsChanged
        )
        rua.disables = [ruv]
        rua.makeConsistent()

        boxAt = gui.widgetBox(self.controlArea, "Classes", addSpace=True)
        gui.checkBox(boxAt, self, 'sortClasses', 'Sort classes',
                     callback=self.optionsChanged)
        gui.separator(boxAt, 2)
        rua = gui.checkBox(
            boxAt, self, "removeClassAttribute",
            "Remove class attribute if there are less than two classes",
            callback=self.removeClassesChanged
        )
        ruv = gui.checkBox(
            gui.indentedBox(boxAt, sep=gui.checkButtonOffsetHint(rua)),
            self,
            "removeClasses",
            "Remove unused class values",
            callback=self.optionsChanged
        )
        rua.disables = [ruv]
        rua.makeConsistent()

        box3 = gui.widgetBox(self.controlArea, 'Statistics', addSpace=True)
        gui.label(box3, self, "Removed attributes: %(removedAttrs)s")
        gui.label(box3, self, "Reduced attributes: %(reducedAttrs)s")
        gui.label(box3, self, "Resorted attributes: %(resortedAttrs)s")
        gui.label(box3, self, "Removed classes: %(removedClasses)s")
        gui.label(box3, self, "Reduced classes: %(reducedClasses)s")
        gui.label(box3, self, "Resorted classes: %(resortedClasses)s")

        gui.auto_commit(self.controlArea, self, "autoSend", "Send Data",
                        checkbox_label="Send automatically",
                        orientation="horizontal")
        gui.rubber(self.controlArea)

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
            self.send("Data", None)
            self.data = None

    def removeAttributesChanged(self):
        if not self.removeAttributes:
            self.preRemoveValues = self.removeValues
            self.removeValues = False
        else:
            self.removeValues = self.preRemoveValues
        self.optionsChanged()

    def removeClassesChanged(self):
        if not self.removeClassAttribute:
            self.preRemoveClasses = self.removeClasses
            self.removeClasses = False
        else:
            self.removeClasses = self.preRemoveClasses
        self.optionsChanged()

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
        remover = Remove(attr_flags, class_flags)
        data = remover(self.data)
        attr_res, class_res = remover.attr_results, remover.class_results

        self.removedAttrs = attr_res['removed']
        self.reducedAttrs = attr_res['reduced']
        self.resortedAttrs = attr_res['sorted']

        self.removedClasses = class_res['removed']
        self.reducedClasses = class_res['reduced']
        self.resortedClasses = class_res['sorted']

        self.send("Data", data)


if __name__ == "__main__":
    appl = QtGui.QApplication([])
    ow = OWPurgeDomain()
    data = Orange.data.Table("car.tab")
    subset = [inst for inst in data
              if inst["buying"] == "v-high"]
    subset = Orange.data.Table(data.domain, subset)
    # The "buying" should be removed and the class "y" reduced
    ow.setData(subset)
    ow.show()
    appl.exec_()
    ow.saveSettings()
