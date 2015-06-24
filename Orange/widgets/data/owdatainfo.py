import threading

from PyQt4 import QtGui, QtCore
from Orange.widgets import widget, gui
from Orange.data.table import Table
from Orange.data import StringVariable, DiscreteVariable, ContinuousVariable
try:
    from Orange.data.sql.table import SqlTable
except ImportError:
    SqlTable = None


class OWDataInfo(widget.OWWidget):
    name = "Data Info"
    id = "orange.widgets.data.info"
    description = """Display the basic information about the data set, like
    number and type of variables in columns and number of rows."""
    icon = "icons/DataInfo.svg"
    author = "Aleš Erjavec, Janez Demsar"
    maintainer_email = "ales.erjavec(@at@)fri.uni-lj.si"
    priority = 80
    category = "Data"
    keywords = ["data", "info"]
    inputs = [("Data", Table, "data")]

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data(None)
        self.data_set_size = self.features = self.meta_attributes = ""
        self.location = ""
        for box in ("Data Set Size", "Features", "Targets", "Meta Attributes",
                    "Location"):
            name = box.lower().replace(" ", "_")
            bo = gui.widgetBox(self.controlArea, box,
                               addSpace=False and box != "Meta Attributes")
            gui.label(bo, self, "%%(%s)s" % name)

        # ensure the widget has some decent minimum width.
        self.targets = "Discrete class with 123 values"
        self.layout().activate()
        # NOTE: The minimum width is set on the 'contained' widget and
        # not `self`. The layout will set a fixed size to `self` taking
        # into account the minimum constraints of the children (it would
        # override any minimum/fixed size set on `self`).
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

        self.targets = ""

    def data(self, data):
        def n_or_none(i):
            return i or "(none)"

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
            return

        sparses = [s for s, m in (("features", data.X_density),
                                  ("meta attributes", data.metas_density),
                                  ("targets", data.Y_density)) if m() > 1]
        if sparses:
            sparses = "<p>Sparse representation: %s</p>" % ", ".join(sparses)
        else:
            sparses = ""
        domain = data.domain
        self.data_set_size = pack_table((
            ("Rows", '~{}'.format(data.approx_len())),
            ("Variables", len(domain)))) + sparses

        def update_size():
            self.data_set_size = pack_table((
                ("Rows", len(data)),
                ("Variables", len(domain)))) + sparses

        threading.Thread(target=update_size).start()

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
            if class_var.is_continuous:
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

        if SqlTable is not None and isinstance(data, SqlTable):
            connection_string = ' '.join(
                '%s=%s' % (key, value)
                for key, value in data.connection_params.items()
                if value is not None)
            self.location = "Table '%s', using connection:\n%s" % (
                data.table_name, connection_string)
        else:
            self.location = "Data is stored in memory"


if __name__ == "__main__":
    a = QtGui.QApplication([])
    ow = OWDataInfo()
    ow.show()
    ow.data(Table("iris"))
    ow.raise_()
    a.exec_()
    ow.saveSettings()
