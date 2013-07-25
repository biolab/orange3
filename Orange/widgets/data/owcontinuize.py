#from orngWrap import PreprocessedLearner
from PyQt4 import QtCore
from PyQt4 import QtGui


from Orange.data.continuizer import DomainContinuizer
from Orange.data.table import Table
from Orange.data.variable import Variable
from Orange.widgets import gui, widget
from Orange.widgets.settings import ContextSetting, ClassValuesContextHandler, Setting

class OWContinuize(widget.OWWidget):
    _name = "Continuize"
    _description = "Turns discrete attributes into continuous and, optionally, normalizes the continuous values."
    _icon = "icons/Continuize.svg"
    _author = "Martin Frlin"
    _category = "Data"
    _keywords = ["data", "continuize"]

    inputs = [("Data", Table, "setData")]
    outputs = [("Data", Table)]

    want_main_area = False

    multinomialTreatment = Setting(0)
    classTreatment = Setting(0)
    zeroBased = Setting(1)
    continuousTreatment = Setting(0)
    autosend = Setting(0)

    settingsHandler = ClassValuesContextHandler()
    targetValue = ContextSetting("")

    multinomialTreats = (("Target or First value as base", DomainContinuizer.LowestIsBase),
                         ("Most frequent value as base", DomainContinuizer.FrequentIsBase),
                         ("One attribute per value", DomainContinuizer.NValues),
                         ("Ignore multinomial attributes", DomainContinuizer.IgnoreMulti),
                         ("Ignore all discrete attributes", DomainContinuizer.Ignore),
                         ("Treat as ordinal", DomainContinuizer.AsOrdinal),
                         ("Divide by number of values", DomainContinuizer.AsNormalizedOrdinal))

    continuousTreats = (("Leave them as they are", DomainContinuizer.Leave),
                        ("Normalize by span", DomainContinuizer.NormalizeBySpan),
                        ("Normalize by variance", DomainContinuizer.NormalizeByVariance))

    classTreats = (("Leave it as it is", DomainContinuizer.Ignore),
                   ("Treat as ordinal", DomainContinuizer.AsOrdinal),
                   ("Divide by number of values", DomainContinuizer.AsNormalizedOrdinal),
                   ("Specified target value", -1))

    valueRanges = ["from -1 to 1", "from 0 to 1"]

    def __init__(self,parent=None, signalManager = None, name = "Continuizer"):
        widget.OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)

        self.targetValue = 0
        self.autosend = 0
        self.dataChanged = False

        bgMultiTreatment = gui.widgetBox(self.controlArea, "Multinomial attributes")
        gui.radioButtonsInBox(bgMultiTreatment, self, "multinomialTreatment", btnLabels=[x[0] for x in self.multinomialTreats], callback=self.sendDataIf)

        self.controlArea.layout().addSpacing(4)

        bgMultiTreatment = gui.widgetBox(self.controlArea, "Continuous attributes")
        gui.radioButtonsInBox(bgMultiTreatment, self, "continuousTreatment", btnLabels=[x[0] for x in self.continuousTreats], callback=self.sendDataIf)

        self.controlArea.layout().addSpacing(4)

        bgClassTreatment = gui.widgetBox(self.controlArea, "Discrete class attribute")
        self.ctreat =gui.radioButtonsInBox(bgClassTreatment, self, "classTreatment", btnLabels=[x[0] for x in self.classTreats], callback=self.sendDataIf)
#        hbox = OWGUI.widgetBox(bgClassTreatment, orientation = "horizontal")
#        OWGUI.separator(hbox, 19, 4)
        hbox = gui.indentedBox(bgClassTreatment, sep=gui.checkButtonOffsetHint(self.ctreat.buttons[-1]), orientation="horizontal")
        self.cbTargetValue = gui.comboBox(hbox, self, "targetValue", label="Target Value ", items=[], orientation="horizontal", callback=self.cbTargetSelected)
        def setEnabled(*args):
            self.cbTargetValue.setEnabled(self.classTreatment == 3)
        #self.connect(self.ctreat.group, SIGNAL("buttonClicked(int)"), setEnabled)
        self.ctreat.group.clicked.connect(setEnabled)
        setEnabled()

        self.controlArea.layout().addSpacing(4)

        zbbox = gui.widgetBox(self.controlArea, "Value range")
        gui.radioButtonsInBox(zbbox, self, "zeroBased", btnLabels=self.valueRanges, callback=self.sendDataIf)

        self.controlArea.layout().addSpacing(4)

        snbox = gui.widgetBox(self.controlArea, "Send data")
        gui.button(snbox, self, "Send data", callback=self.sendData, default=True)
        gui.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto)
        self.data = None
        self.sendPreprocessor()
        self.resize(150,300)
        #self.adjustSize()

    def cbTargetSelected(self):
        self.classTreatment = 3
        self.sendDataIf()

    def setData(self,data):
        self.closeContext()

        if not data:
            self.data = None
            self.cbTargetValue.clear()
            self.openContext("", self.data)
            self.send("Data", None)
        else:
            if not self.data or data.domain.class_var != self.data.domain.class_var:
                self.cbTargetValue.clear()
                if data.domain.class_var and data.domain.class_var.var_type == Variable.VarTypes.Discrete:
                    for v in data.domain.class_var.values:
                        self.cbTargetValue.addItem(" "+v)
                    self.ctreat.setDisabled(False)
                    self.targetValue = 0
                else:
                    self.ctreat.setDisabled(True)
            self.data = data
            self.openContext("", self.data)
            self.sendData()

    def sendDataIf(self):
        self.dataChanged = True
        if self.autosend:
            self.sendPreprocessor()
            self.sendData()

    def enableAuto(self):
        if self.dataChanged:
            self.sendPreprocessor()
            self.sendData()

    def constructContinuizer(self):
        conzer = DomainContinuizer()
        conzer.zeroBased = self.zeroBased
        conzer.continuousTreatment = self.continuousTreats[self.continuousTreatment][1]
        conzer.multinomialTreatment = self.multinomialTreats[self.multinomialTreatment][1]
        conzer.classTreatment = self.classTreats[self.classTreatment][1]
        return conzer

    def sendPreprocessor(self):
        continuizer = self.constructContinuizer()
        #TODO
        #self.send("Preprocessor", PreprocessedLearner(
         #   lambda data, weightId=0, tc=(self.targetValue if self.classTreatment else -1):
          #      Table(continuizer(data, weightId, tc)
           #           if data.domain.class_var and self.data.domain.class_var.var_type == Variable.VarTypes.Discrete
            #          else continuizer(data, weightId), data)))


    def sendData(self):
        continuizer = self.constructContinuizer()
        if self.data:
            if self.data.domain.class_var.var_type and self.data.domain.class_var.var_type == Variable.VarTypes.Discrete:
                domain = continuizer(self.data, 0, self.targetValue if self.classTreatment else -1)
            else:
                domain = continuizer(self.data, 0)
            domain.addmetas(self.data.domain.metas)
            self.send("Data", Table(domain, self.data))
        self.dataChanged = False

    def sendReport(self):
        self.reportData(self.data, "Input data")
        clstr = "None"
        if self.data is not None:
            classVar = self.data.domain.class_var
            if self.classTreatment == 3 and classVar and classVar.var_type == Variable.VarTypes.Discrete and len(classVar.values) >= 2:
                clstr = "Dummy variable for target '%s'" % classVar.values[self.targetValue]
            else:
                clstr = self.classTreats[self.classTreatment][0]
        self.reportSettings("Settings",
                            [("Multinominal attributes", self.multinomialTreats[self.multinomialTreatment][0]),
                             ("Continuous attributes", self.continuousTreats[self.continuousTreatment][0]),
                             ("Class attribute", clstr),
                             ("Value range", self.valueRanges[self.zeroBased])])

if __name__ == "__main__":
    import sys
    a = QtGui.QApplication(sys.argv)
    ow = OWContinuize()
    #data = orange.ExampleTable("d:\\ai\\orange\\test\\iris")
#    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    data = Table("../../doc/datasets/iris.tab")
    ow.setData(data)
    ow.show()
    a.exec_()
    ow.saveSettings()
