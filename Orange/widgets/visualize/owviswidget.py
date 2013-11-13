import os

from Orange.canvas.utils import environ
from PyQt4.QtGui import QListWidget, QIcon, QSizePolicy
from Orange.widgets import gui as OWGUI
from Orange.widgets.widget import OWWidget


class OWVisWidget(OWWidget):
    def createShowHiddenLists(self, placementTab, callback=None):
        maxWidth = 180
        self.updateCallbackFunction = callback
        self.shownAttributes = []
        self.selectedShown = []
        self.hiddenAttributes = []
        self.selectedHidden = []

        self.shownAttribsGroup = OWGUI.widgetBox(placementTab, " Shown attributes ")
        self.addRemoveGroup = OWGUI.widgetBox(placementTab, 1, orientation="horizontal")
        self.hiddenAttribsGroup = OWGUI.widgetBox(placementTab, " Hidden attributes ")

        hbox = OWGUI.widgetBox(self.shownAttribsGroup, orientation='horizontal')
        self.shownAttribsLB = OWGUI.listBox(hbox, self, "selectedShown", "shownAttributes",
                                            callback=self.resetAttrManipulation, dragDropCallback=callback,
                                            enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)
        #self.shownAttribsLB.setMaximumWidth(maxWidth)
        vbox = OWGUI.widgetBox(hbox, orientation='vertical')
        self.buttonUPAttr = OWGUI.button(vbox, self, "", callback=self.moveAttrUP,
                                         tooltip="Move selected attributes up")
        self.buttonDOWNAttr = OWGUI.button(vbox, self, "", callback=self.moveAttrDOWN,
                                           tooltip="Move selected attributes down")
        self.buttonUPAttr.setIcon(QIcon(os.path.join(environ.widget_install_dir, "icons/Dlg_up3.png")))
        self.buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.buttonUPAttr.setMaximumWidth(30)
        self.buttonDOWNAttr.setIcon(QIcon(os.path.join(environ.widget_install_dir, "icons/Dlg_down3.png")))
        self.buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.buttonDOWNAttr.setMaximumWidth(30)

        self.attrAddButton = OWGUI.button(self.addRemoveGroup, self, "", callback=self.addAttribute,
                                          tooltip="Add (show) selected attributes")
        self.attrAddButton.setIcon(QIcon(os.path.join(environ.widget_install_dir, "icons/Dlg_up3.png")))
        self.attrRemoveButton = OWGUI.button(self.addRemoveGroup, self, "", callback=self.removeAttribute,
                                             tooltip="Remove (hide) selected attributes")
        self.attrRemoveButton.setIcon(QIcon(os.path.join(environ.widget_install_dir, "icons/Dlg_down3.png")))
        self.showAllCB = OWGUI.checkBox(self.addRemoveGroup, self, "show_all_attributes", "Show all",
                                        callback=self.cbShowAllAttributes)

        self.hiddenAttribsLB = OWGUI.listBox(self.hiddenAttribsGroup, self, "selectedHidden", "hiddenAttributes",
                                             callback=self.resetAttrManipulation, dragDropCallback=callback,
                                             enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)
        #self.hiddenAttribsLB.setMaximumWidth(maxWidth + 27)


    def getDataDomain(self):
    #        if hasattr(self, "graph") and hasattr(self.graph, "dataDomain"):
    #            return self.graph.dataDomain
        if hasattr(self, "data") and self.data:
            return self.data.domain
        else:
            return None

    def resetAttrManipulation(self):
        if self.selectedShown:
            mini, maxi = min(self.selectedShown), max(self.selectedShown)
            tightSelection = maxi - mini == len(self.selectedShown) - 1
        self.buttonUPAttr.setEnabled(self.selectedShown != [] and tightSelection and mini)
        self.buttonDOWNAttr.setEnabled(
            self.selectedShown != [] and tightSelection and maxi < len(self.shownAttributes) - 1)
        self.attrAddButton.setDisabled(not self.selectedHidden or self.show_all_attributes)
        self.attrRemoveButton.setDisabled(not self.selectedShown or self.show_all_attributes)
        domain = self.getDataDomain()
        if domain and self.hiddenAttributes and domain.class_var and self.hiddenAttributes[0][0] != domain.class_var.name:
            self.showAllCB.setChecked(0)


    def moveAttrSelection(self, labels, selection, dir):
        if hasattr(self, "graph"):
            self.graph.insideColors = None
            self.graph.clusterClosure = None
            self.graph.potentialsBmp = None

        labs = getattr(self, labels)
        sel = getattr(self, selection)
        mini, maxi = min(sel), max(sel) + 1
        if dir == -1:
            setattr(self, labels, labs[:mini - 1] + labs[mini:maxi] + [labs[mini - 1]] + labs[maxi:])
        else:
            setattr(self, labels, labs[:mini] + [labs[maxi]] + labs[mini:maxi] + labs[maxi + 1:])
        setattr(self, selection, [x + dir for x in sel])

        self.resetAttrManipulation()
        if hasattr(self, "sendShownAttributes"):
            self.sendShownAttributes()
        self.graph.potentialsBmp = None
        if self.updateCallbackFunction:
            self.updateCallbackFunction()
        if hasattr(self, "graph"):
            self.graph.removeAllSelections()

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        self.moveAttrSelection("shownAttributes", "selectedShown", -1)

    # move selected attribute in "Attribute Order" list one place down
    def moveAttrDOWN(self):
        self.moveAttrSelection("shownAttributes", "selectedShown", 1)


    def cbShowAllAttributes(self):
        if self.show_all_attributes:
            self.addAttribute(True)
        self.resetAttrManipulation()

    def addAttribute(self, addAll=False):
        if hasattr(self, "graph"):
            self.graph.insideColors = None
            self.graph.clusterClosure = None

        if addAll:
            self.setShownAttributeList()
        else:
            self.setShownAttributeList(self.shownAttributes + [self.hiddenAttributes[i] for i in self.selectedHidden])
        self.selectedHidden = []
        self.selectedShown = []
        self.resetAttrManipulation()

        if hasattr(self, "sendShownAttributes"):
            self.sendShownAttributes()
        if self.updateCallbackFunction:
            self.updateCallbackFunction()
        if hasattr(self, "graph"):
            self.graph.removeAllSelections()

    def removeAttribute(self):
        if hasattr(self, "graph"):
            self.graph.insideColors = None
            self.graph.clusterClosure = None

        newShown = self.shownAttributes[:]
        self.selectedShown.sort(reverse=True)
        for i in self.selectedShown:
            del newShown[i]
        self.setShownAttributeList(newShown)

        if hasattr(self, "sendShownAttributes"):
            self.sendShownAttributes()
        if self.updateCallbackFunction:
            self.updateCallbackFunction()
        if hasattr(self, "graph"):
            self.graph.removeAllSelections()

    def getShownAttributeList(self):
        return [a[0] for a in self.shownAttributes]


    def setShownAttributeList(self, shownAttributes=None):
        shown = []
        hidden = []

        domain = self.getDataDomain()
        if domain:
            if shownAttributes:
                if type(shownAttributes[0]) == tuple:
                    shown = shownAttributes
                else:
                    shown = [(domain[a].name, domain[a].var_type) for a in shownAttributes]
                hidden = [x for x in [(a.name, a.var_type) for a in domain.attributes] if x not in shown]
            else:
                shown = [(a.name, a.var_type) for a in domain.attributes]
                if not self.show_all_attributes:
                    hidden = shown[10:]
                    shown = shown[:10]

            if domain.class_var and (domain.class_var.name, domain.class_var.var_type) not in shown:
                hidden += [(domain.class_var.name, domain.class_var.var_type)]

        self.shownAttributes = shown
        self.hiddenAttributes = hidden
        self.selectedHidden = []
        self.selectedShown = []
        self.resetAttrManipulation()

        if hasattr(self, "sendShownAttributes"):
            self.sendShownAttributes()

