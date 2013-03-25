from PyQt4 import QtCore
from PyQt4 import QtGui
from Orange.widgets import widget
from Orange.data import Table
from Orange.data.sql.table import SqlTable

class OWSqlIris(widget.OWWidget):
    _name = "SQL iris"
    _id = "orange.widgets.data.sqliris"
    _description = """
    Load iris dataset from SQL."""
    _long_description = """
    Dummy widget that reads the data from iris table in database test
    on the localhost, requireing no authentication."""
    _icon = "icons/File.svg"
    _author = "Anze Staric"
    _maintainer_email = "anze.staric@fri.uni-lj.si"
    _priority = 10
    _category = "Data"
    _keywords = ["data", "file", "load", "read"]
    outputs = [{"name": "Data",
                "type": Table,
                "doc": "Attribute-valued data set read from the input file."}]

    want_main_area = False

    def __init__(self, parent=None, signalManager=None, stored_settings=None):
        super(OWSqlIris, self).__init__(parent=parent,
                                        signalManager=signalManager,
                                        stored_settings=stored_settings)
        table = SqlTable("pgsql://localhost/test/iris")
        self.send("Data", table)