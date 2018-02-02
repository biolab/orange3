from collections import OrderedDict
import threading

from AnyQt import QtWidgets
from AnyQt import QtCore

from Orange.widgets import widget, gui
from Orange.widgets.widget import Input
from Orange.data.table import Table
from Orange.data import StringVariable, DiscreteVariable, ContinuousVariable
from Orange.canvas import report
try:
    from Orange.data.sql.table import SqlTable
except ImportError:
    SqlTable = None


class OWDataInfo(widget.OWWidget):
    name = "Data Info"
    id = "orange.widgets.data.info"
    description = """Display basic information about the dataset, such
    as the number and type of variables in the columns and the number of rows."""
    icon = "icons/DataInfo.svg"
    priority = 80
    category = "Data"
    keywords = ["data", "info"]

    class Inputs:
        data = Input("Data", Table)

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data(None)
        self.data_set_size = self.features = self.meta_attributes = ""
        self.location = ""
        for box in ("Data Set Size", "Features", "Targets", "Meta Attributes",
                    "Location"):
            name = box.lower().replace(" ", "_")
            bo = gui.vBox(self.controlArea, box,
                          addSpace=False and box != "Meta Attributes")
            gui.label(bo, self, "%%(%s)s" % name)

        # ensure the widget has some decent minimum width.
        self.targets = "Discrete outcome with 123 values"
        self.layout().activate()
        # NOTE: The minimum width is set on the 'contained' widget and
        # not `self`. The layout will set a fixed size to `self` taking
        # into account the minimum constraints of the children (it would
        # override any minimum/fixed size set on `self`).
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        self.targets = ""
        self.data_desc = None

    @Inputs.data
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
            self.data_desc = None
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
            disc_features = count(domain.attributes, DiscreteVariable)
            cont_features = count(domain.attributes, ContinuousVariable)
            self.features = pack_table((
                ("Discrete", n_or_none(disc_features)),
                ("Numeric", n_or_none(cont_features))
            ))

        if not domain.metas:
            self.meta_attributes = "None"
        else:
            disc_metas = count(domain.metas, DiscreteVariable)
            cont_metas = count(domain.metas, ContinuousVariable)
            str_metas = count(domain.metas, StringVariable)
            self.meta_attributes = pack_table((
                ("Discrete", n_or_none(disc_metas)),
                ("Numeric", n_or_none(cont_metas)),
                ("Textual", n_or_none(str_metas))))

        class_var = domain.class_var
        if class_var:
            if class_var.is_continuous:
                self.targets = "Numeric target variable"
            else:
                self.targets = "Discrete outcome with %i values" % \
                               len(class_var.values)
        elif domain.class_vars:
            disc_class = count(domain.class_vars, DiscreteVariable)
            cont_class = count(domain.class_vars, ContinuousVariable)
            if not cont_class:
                self.targets = "Multitarget data,\n%i categorical targets" % \
                               n_or_none(disc_class)
            elif not disc_class:
                self.targets = "Multitarget data,\n%i numeric targets" % \
                               n_or_none(cont_class)
            else:
                self.targets = "<p>Multi target data</p>\n" + pack_table(
                    (("Categorical", disc_class), ("Numeric", cont_class)))

        self.data_desc = dd = OrderedDict()

        if SqlTable is not None and isinstance(data, SqlTable):
            connection_string = ' '.join(
                '%s=%s' % (key, value)
                for key, value in data.connection_params.items()
                if value is not None and key != 'password')
            self.location = "Table '%s', using connection:\n%s" % (
                data.table_name, connection_string)
            dd["Rows"] = data.approx_len()
        else:
            self.location = "Data is stored in memory"
            dd["Rows"] = len(data)

        def join_if(items):
            return ", ".join(s.format(n) for s, n in items if n)

        dd["Features"] = len(domain.attributes) and join_if((
            ("{} categorical", disc_features),
            ("{} numeric", cont_features)
        ))
        if domain.class_var:
            name = domain.class_var.name
            if domain.class_var.is_discrete:
                dd["Target"] = "categorical outcome '{}'".format(name)
            else:
                dd["Target"] = "numeric target '{}'".format(name)
        elif domain.class_vars:
            tt = ""
            if disc_class:
                tt += report.plural("{number} categorical outcome{s}", disc_class)
            if cont_class:
                tt += report.plural("{number} numeric target{s}", cont_class)
        dd["Meta attributes"] = len(domain.metas) > 0 and join_if((
            ("{} categorical", disc_metas),
            ("{} numeric", cont_metas),
            ("{} textual", str_metas)
        ))

    def send_report(self):
        if self.data_desc:
            self.report_items(self.data_desc)

if __name__ == "__main__":
    a = QtWidgets.QApplication([])
    ow = OWDataInfo()
    ow.show()
    ow.data(Table("iris"))
    ow.raise_()
    a.exec_()
    ow.saveSettings()
