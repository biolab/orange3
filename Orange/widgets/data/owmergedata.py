import math
import itertools

from PyQt4.QtCore import Qt
from collections import defaultdict

from PyQt4 import QtGui
import numpy

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.data import Table


INSTANCEID = "Source position (index)"
INDEX = "Position (index)"


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected data features."
    icon = "icons/MergeData.svg"
    priority = 1110

    inputs = [("Data A", Orange.data.TableBase, "setDataA", widget.Default),
              ("Data B", Orange.data.TableBase, "setDataB")]
    outputs = [("Merged Data A+B", Orange.data.TableBase, ),
               ("Merged Data B+A", Orange.data.TableBase, )]

    attr_a = settings.Setting('', schema_only=True)
    attr_b = settings.Setting('', schema_only=True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        # data
        self.dataA = None
        self.dataB = None

        # GUI
        w = QtGui.QWidget(self)
        self.controlArea.layout().addWidget(w)
        grid = QtGui.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        w.setLayout(grid)

        # attribute A selection
        boxAttrA = gui.vBox(self, self.tr("Attribute A"), addToLayout=False)
        grid.addWidget(boxAttrA, 0, 0)

        self.attrViewA = gui.comboBox(boxAttrA, self, 'attr_a',
                                      orientation=Qt.Horizontal,
                                      sendSelectedValue=True,
                                      callback=self._invalidate)
        self.attrModelA = itemmodels.VariableListModel()
        self.attrViewA.setModel(self.attrModelA)

        # attribute  B selection
        boxAttrB = gui.vBox(self, self.tr("Attribute B"), addToLayout=False)
        grid.addWidget(boxAttrB, 0, 1)

        self.attrViewB = gui.comboBox(boxAttrB, self, 'attr_b',
                                      orientation=Qt.Horizontal,
                                      sendSelectedValue=True,
                                      callback=self._invalidate)
        self.attrModelB = itemmodels.VariableListModel()
        self.attrViewB.setModel(self.attrModelB)

        # info A
        boxDataA = gui.vBox(self, self.tr("Data A Input"), addToLayout=False)
        grid.addWidget(boxDataA, 1, 0)
        self.infoBoxDataA = gui.widgetLabel(boxDataA, self.dataInfoText(None))

        # info B
        boxDataB = gui.vBox(self, self.tr("Data B Input"), addToLayout=False)
        grid.addWidget(boxDataB, 1, 1)
        self.infoBoxDataB = gui.widgetLabel(boxDataB, self.dataInfoText(None))

        gui.rubber(self)

    def _setAttrs(self, model, data, othermodel, otherdata):
        model[:] = allvars(data) if data is not None else []

        if data is not None and otherdata is not None and \
                len(numpy.intersect1d(data.index, otherdata.index)):
            for model_ in (model, othermodel):
                if len(model_) and model_[0] != INSTANCEID:
                    model_.insert(0, INSTANCEID)

    @check_sql_input
    def setDataA(self, data):
        self.dataA = data
        self._setAttrs(self.attrModelA, data, self.attrModelB, self.dataB)
        curr_index = -1
        if self.attr_a:
            curr_index = next((i for i, val in enumerate(self.attrModelA)
                               if str(val) == self.attr_a), -1)
        if curr_index != -1:
            self.attrViewA.setCurrentIndex(curr_index)
        else:
            self.attr_a = INDEX
        self.infoBoxDataA.setText(self.dataInfoText(data))

    @check_sql_input
    def setDataB(self, data):
        self.dataB = data
        self._setAttrs(self.attrModelB, data, self.attrModelA, self.dataA)
        curr_index = -1
        if self.attr_b:
            curr_index = next((i for i, val in enumerate(self.attrModelB)
                               if str(val) == self.attr_b), -1)
        if curr_index != -1:
            self.attrViewB.setCurrentIndex(curr_index)
        else:
            self.attr_b = INDEX
        self.infoBoxDataB.setText(self.dataInfoText(data))

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
        ninstances = 0
        nvariables = 0
        if data is not None:
            ninstances = len(data)
            nvariables = len(data.domain)

        instances = self.tr("%n instance(s)", None, ninstances)
        attributes = self.tr("%n variable(s)", None, nvariables)
        return "\n".join([instances, attributes])

    def commit(self):
        AB, BA = None, None
        if self.attr_a and self.attr_b and self.dataA is not None and self.dataB is not None:
            varA = (self.attr_a if self.attr_a in (INDEX, INSTANCEID) else
                    self.dataA.domain[self.attr_a])
            varB = (self.attr_b if self.attr_b in (INDEX, INSTANCEID) else
                    self.dataB.domain[self.attr_b])
            if varA == INDEX or varB == INDEX:
                # temporarily ignore index, this is bad, mmkay?
                ai = self.dataA.index
                bi = self.dataB.index
                self.dataA.reset_index(drop=True, inplace=True)
                self.dataB.reset_index(drop=True, inplace=True)
                AB = self.dataA.merge(self.dataB, left_index=True, right_index=True)
                BA = self.dataB.merge(self.dataA, left_index=True, right_index=True)
                self.dataA.index = ai
                self.dataB.index = bi
            elif varB == INSTANCEID or varB == INSTANCEID:
                AB = self.dataA.merge(self.dataB, left_index=True, right_index=True)
                BA = self.dataB.merge(self.dataA, left_index=True, right_index=True)
            else:
                AB = self.dataA.merge(self.dataB, left_on=varA, right_on=varB)
                BA = self.dataB.merge(self.dataA, left_on=varB, right_on=varA)
        self.send("Merged Data A+B", AB)
        self.send("Merged Data B+A", BA)

    def _invalidate(self):
        self.commit()

    def send_report(self):
        attr_a = None
        attr_b = None
        if self.dataA is not None:
            attr_a = self.attr_a
            if attr_a in self.dataA.domain:
                attr_a = self.dataA.domain[attr_a]
        if self.dataB is not None:
            attr_b = self.attr_b
            if attr_b in self.dataB.domain:
                attr_b = self.dataB.domain[attr_b]
        self.report_items((
            ("Attribute A", attr_a),
            ("Attribute B", attr_b),
        ))


def allvars(data):
    return (INDEX,) + data.domain.attributes + data.domain.class_vars + data.domain.metas


def test():
    app = QtGui.QApplication([])

    w = OWMergeData()
    zoo = Orange.data.Table("zoo")
    A = zoo[:, [0, 1, 2, "type", -1]]
    B = zoo[:, [3, 4, 5, "type", -1]]
    w.setDataA(A)
    w.setDataB(B)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
