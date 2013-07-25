from PyQt4 import QtGui

from Orange.data.domain import Domain
from Orange.data.table import Table
from Orange.data.variable import Variable
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting

class OWPurgeDomain(widget.OWWidget):
    _name = "Purge Domain"
    _description = "Removes redundant values and attributes, sorts values."
    _icon = "icons/PurgeDomain.svg"
    _author = "Martin Frlin"
    _category = "Data"
    _keywords = ["data", "purge", "domain"]

    inputs = [("Data", Table, "setData")]
    outputs = [("Data", Table)]

    removeValues = Setting(1)
    removeAttributes = Setting(1)
    removeClassAttribute = Setting(1)
    removeClasses = Setting(1)
    autoSend = Setting(1)
    sortValues = Setting(True)
    sortClasses = Setting(True)

    def __init__(self, parent=None, signalManager=None):
        widget.OWWidget.__init__(self, parent, signalManager, 'PurgeDomain', wantMainArea=False)
        self.data = None

        self.preRemoveValues = 1
        self.preRemoveClasses = 1
        self.autoSend = 1
        self.dataChanged = False

        self.removedAttrs = self.reducedAttrs = self.resortedAttrs = self.classAttr = "-"

        boxAt =gui.widgetBox(self.controlArea, "Attributes", addSpace=True)
        gui.checkBox(boxAt, self, 'sortValues', 'Sort attribute values', callback = self.optionsChanged)
        rua = gui.checkBox(boxAt, self, "removeAttributes", "Remove attributes with less than two values", callback = self.removeAttributesChanged)

        ruv = gui.checkBox(gui.indentedBox(boxAt, sep=gui.checkButtonOffsetHint(rua)), self, "removeValues", "Remove unused attribute values", callback = self.optionsChanged)
        rua.disables = [ruv]
        rua.makeConsistent()


        boxAt = gui.widgetBox(self.controlArea, "Classes", addSpace=True)
        gui.checkBox(boxAt, self, 'sortClasses', 'Sort classes', callback = self.optionsChanged)
        rua = gui.checkBox(boxAt, self, "removeClassAttribute", "Remove class attribute if there are less than two classes", callback = self.removeClassesChanged)
        ruv = gui.checkBox(gui.indentedBox(boxAt, sep=gui.checkButtonOffsetHint(rua)), self, "removeClasses", "Remove unused class values", callback = self.optionsChanged)
        rua.disables = [ruv]
        rua.makeConsistent()


        box3 = gui.widgetBox(self.controlArea, 'Statistics', addSpace=True)
        gui.label(box3, self, "Removed attributes: %(removedAttrs)s")
        gui.label(box3, self, "Reduced attributes: %(reducedAttrs)s")
        gui.label(box3, self, "Resorted attributes: %(resortedAttrs)s")
        gui.label(box3, self, "Class attribute: %(classAttr)s")

        box2 = gui.widgetBox(self.controlArea, "Send")
        btSend = gui.button(box2, self, "Send data", callback = self.process, default=True)
        cbAutoSend = gui.checkBox(box2, self, "autoSend", "Send automatically")

        gui.setStopper(self, btSend, cbAutoSend, "dataChanged", self.process)

        gui.rubber(self.controlArea)

#        OWGUI.separator(self.controlArea, height=24)

        #self.adjustSize()

    def setData(self, dataset):
        if dataset:
            self.data = dataset
            self.process()
        else:
            self.reducedAttrs = self.removedAttrs = self.resortedAttrs = self.classAttr = ""
            self.send("Data", None)
            self.data = None
        self.dataChanged = False

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
        if self.autoSend:
            self.process()
        else:
            self.dataChanged = True

    def sortAttrValues(self, attr, interattr=None):
        if not interattr:
            interattr = attr

        newvalues = list(interattr.values)
        newvalues.sort()
        if newvalues == list(interattr.values):
            return interattr

        newattr = orange.EnumVariable(interattr.name, values=newvalues)
        newattr.getValueFrom = orange.ClassifierByLookupTable(newattr, attr)
        lookupTable = newattr.getValueFrom.lookupTable
        distributions = newattr.getValueFrom.distributions
        for val in interattr.values:
            idx = attr.values.index(val)
            lookupTable[idx] = val
            distributions[idx][newvalues.index(val)] += 1
        return newattr

    def process(self):
        if self.data == None:
            return

        self.reducedAttrs = 0
        self.removedAttrs = 0
        self.resortedAttrs = 0
        self.classAttribute = 0

        if self.removeAttributes or self.sortValues:
            newattrs = []
            for attr in self.data.domain.attributes:
                if attr.varType == Variable.VarTypes.Continuous:
                    if orange.RemoveRedundantOneValue.has_at_least_two_values(self.data, attr):
                        newattrs.append(attr)
                    else:
                        self.removedAttrs += 1
                    continue

                if attr.varType != Variable.VarTypes.Discrete:
                    newattrs.append(attr)
                    continue

                if self.removeValues:
                    newattr = orange.RemoveUnusedValues(attr, self.data)
                    if not newattr:
                        self.removedAttrs += 1
                        continue

                    if newattr != attr:
                        self.reducedAttrs += 1
                else:
                    newattr = attr

                if self.removeValues and len(newattr.values) < 2:
                    self.removedAttrs += 1
                    continue

                if self.sortValues:
                    newnewattr = self.sortAttrValues(attr, newattr)
                    if newnewattr != newattr:
                        self.resortedAttrs += 1
                        newattr = newnewattr

                newattrs.append(newattr)
        else:
            newattrs = self.data.domain.attributes


        klass = self.data.domain.classVar
        classChanged = False
        if not klass:
            newclass = klass
            self.classAttr = "No class"
        elif klass.varType != Variable.VarTypes.Discrete:
            newclass = klass
            self.classAttr = "Class is not discrete"
        elif not (self.removeClassAttribute or self.sortClasses):
            newclass = klass
            self.classAttr = "Class is not checked"
        else:
            self.classAttr = ""

            if self.removeClasses:
                newclass = orange.RemoveUnusedValues(klass, self.data)
            else:
                newclass = klass

            if not newclass or self.removeClassAttribute and len(newclass.values) < 2:
                newclass = None
                self.classAttr = "Class is removed"
            elif len(newclass.values) != len(klass.values):
                    self.classAttr = "Class is reduced"

            if newclass and self.sortClasses:
                newnewclass = self.sortAttrValues(klass, newclass)
                if newnewclass != newclass:
                    if self.classAttr:
                        self.classAttr = "Class is reduced and sorted"
                    else:
                        self.classAttr = "Class is sorted"
                    newclass = newnewclass

            if not self.classAttr:
                self.classAttr = "Class is unchanged"

        if self.reducedAttrs or self.removedAttrs or self.resortedAttrs or newclass != klass:
            newDomain = Domain(newattrs, newclass)
            newData = Table(newDomain, self.data)
        else:
            newData = self.data

        self.send("Data", newData)

        self.dataChanged = False


if __name__=="__main__":
    import sys
    appl = QtGui.QApplication(sys.argv)
    ow = OWPurgeDomain()
    #data = orange.ExampleTable('..\\..\\doc\\datasets\\car.tab')
    #data.domain.attributes[3].values.append("X")
    #ow.setData(data)
    ow.show()
    appl.exec_()
    ow.saveSettings()
