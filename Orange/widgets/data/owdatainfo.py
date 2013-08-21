from PyQt4 import QtGui, QtCore
from Orange.widgets import widget, gui
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable
from Orange.data import StringVariable, DiscreteVariable, ContinuousVariable


class OWDataInfo(widget.OWWidget):
    _name = "Data Info"
    _id = "orange.widgets.data.info"
    _description = "Shows basic information about the data"
    _long_description = """Shows the number of data instances and number of
    variables of different types and kinds."""
    _icon = "icons/DataInfo.svg"
    _author = "Aleš Erjavec, Janez Demsar"
    _maintainer_email = "ales.erjavec(@at@)fri.uni-lj.si"
    _priority = 80
    _category = "Data"
    _keywords = ["data", "info"]
    inputs = [("Data", Table, "data")]

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data(None)
        for box in ("Data Set Size", "Features", "Targets", "Meta Attributes",
                    "Location"):
            name = box.lower().replace(" ", "_")
            bo = gui.widgetBox(self.controlArea, box,
                               addSpace=box != "Meta Attributes")
            gui.label(bo, self, "%%(%s)s" % name)

    def resize(self):
        QtGui.qApp.processEvents()
        QtCore.QTimer.singleShot(0, self.adjustSize)

    def data(self, data):
        def n_or_none(i):
            return i or "-"

        def count(s, tpe):
            return sum(isinstance(x, tpe) for x in s)

        def count_n(s, tpe):
            return n_or_none(count(s, tpe))

        def pack_table(data):
            return "<table>\n" + "\n".join(
                '<tr><td align="right" width="90">%s:</td>\n'
                '<td width="40">%s</td></tr>\n' % dv for dv in data
            ) + "</table>\n"

        if data is None:
            self.data_set_size = "No data"
            self.features = self.targets = self.meta_attributes = "None"
            self.location = ""
            self.resize()
            return

        sparses = [s for s, m in (("features", data.X_density),
                                  ("meta attributes", data.metas_density),
                                  ("targets", data.Y_density)) if m() > 1]
        if sparses:
            sparses = "<p>Sparse representation: %s</p>" % ", ".join(sparses)
        else:
            sparses = ""
        domain = data.domain
        self.data_set_size = pack_table((("Samples", len(data)),
                                         ("Variables", len(domain)))) + sparses

        if not domain.attributes:
            self.features = "None"
        else:
            self.features = pack_table((
                ("Discrete", count_n(domain.attributes, DiscreteVariable)),
                ("Continuous", count_n(domain.attributes, ContinuousVariable)))
            )

        if not domain.metas:
            self.meta_attributes = "None"
        else:
            self.meta_attributes = pack_table((
                ("Discrete", count_n(domain.metas, DiscreteVariable)),
                ("Continuous", count_n(domain.metas, ContinuousVariable)),
                ("String", count_n(domain.metas, StringVariable))))

        class_var = domain.class_var
        if class_var:
            if isinstance(class_var, ContinuousVariable):
                self.targets = "Continuous target variable"
            else:
                self.targets = "Discrete class with %i values" % \
                               len(class_var.values)
        elif domain.class_vars:
            dis = count_n(domain.class_vars, DiscreteVariable)
            con = count_n(domain.class_vars, ContinuousVariable)
            if not con:
                self.targets = "Multitarget data,\n%i discrete targets" % dis
            elif not dis:
                self.targets = "Multitarget data,\n%i continuous targets" % con
            else:
                self.targets = "<p>Multi target data</p>\n" + pack_table(
                    (("Discrete", dis), ("Continuous", con)))

        if isinstance(data, SqlTable):
            self.location = "Table '%s' in database '%s/%s'" % (
                data.name, data.host, data.database)
        else:
            self.location = "Data is stored in memory"

        self.resize()

if __name__ == "__main__":
    a = QtGui.QApplication([])
    ow = OWDataInfo()
    ow.show()
    ow.data(Table("iris"))
    a.exec_()
    ow.saveSettings()
