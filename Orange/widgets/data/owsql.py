import sys
import psycopg2

from PyQt4 import QtGui

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting

from Orange.data import Table
from Orange.data.sql.table import SqlTable


class OWSql(widget.OWWidget):
    _name = "SQL Table"
    _id = "orange.widgets.data.sql"
    _description = """
    Load dataset from SQL."""
    _long_description = """
    Sql widget connects to server and opens data from there. """
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

    host = Setting(None)
    database = Setting(None)
    username = Setting(None)
    password = Setting(None)
    table = Setting(None)
    tables = Setting([])

    def __init__(self, parent=None, signalManager=None, stored_settings=None):
        super(OWSql, self).__init__(parent=parent,
                                    signalManager=signalManager,
                                    stored_settings=stored_settings)

        self._connection = None

        vbox = gui.widgetBox(self.controlArea, "Server", addSpace=True)
        box = gui.widgetBox(vbox)
        self.servertext = QtGui.QLineEdit(box)
        self.servertext.setPlaceholderText('Server')
        if self.host:
            self.servertext.setText(self.host)
        box.layout().addWidget(self.servertext)
        self.databasetext = QtGui.QLineEdit(box)
        self.databasetext.setPlaceholderText('Database')
        if self.database:
            self.databasetext.setText(self.database)
        box.layout().addWidget(self.databasetext)
        self.usernametext = QtGui.QLineEdit(box)
        self.usernametext.setPlaceholderText('Username')
        if self.username:
            self.usernametext.setText(self.username)
        box.layout().addWidget(self.usernametext)
        self.passwordtext = QtGui.QLineEdit(box)
        self.passwordtext.setPlaceholderText('Password')
        self.passwordtext.setEchoMode(QtGui.QLineEdit.Password)
        if self.password:
            self.passwordtext.setText(self.password)
        box.layout().addWidget(self.passwordtext)

        tables = gui.widgetBox(box, orientation='horizontal')
        self.tablecombo = QtGui.QComboBox(tables)
        for i, item in enumerate(['Select a table'] + self.tables):
            self.tablecombo.addItem(item)
            if item == self.table:
                self.tablecombo.setCurrentIndex(i)

        tables.layout().addWidget(self.tablecombo)
        self.tablecombo.activated[int].connect(self.open_table)
        self.connectbutton = gui.button(
            tables, self, '↻', callback=self.connect)
        self.connectbutton.setSizePolicy(
            QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        tables.layout().addWidget(self.connectbutton)

        if self.table:
            self.open_table()

    def connect(self):
        self.host = self.servertext.text()
        self.database = self.databasetext.text()
        self.username = self.usernametext.text() or None
        self.password = self.passwordtext.text() or None
        self._connection = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.username,
            password=self.password
        )
        print("Connected")

        self.refresh_tables()

    def refresh_tables(self):
        if self._connection is None:
            return
        cur = self._connection.cursor()
        cur.execute("SELECT table_name "
                    "  FROM information_schema.tables "
                    " WHERE table_schema = 'public'")
        self.tablecombo.clear()
        self.tablecombo.addItem("Select a table")
        tables = []
        for table_name, in cur.fetchall():
            self.tablecombo.addItem(table_name)
            tables.append(table_name)
        self.tables = tables

    def open_table(self):
        if self.tablecombo.currentIndex() == 0:
            return

        self.table = self.tablecombo.currentText()

        table = SqlTable(host=self.host,
                         database=self.database,
                         user=self.username,
                         password=self.password,
                         table=self.table)
        print("Created table")
        self.send("Data", table)


if __name__ == "__main__":
    import os
    a = QtGui.QApplication(sys.argv)
    settings = os.path.join(widget.environ.widget_settings_dir,
                          OWSql._name + ".ini")
    ow = OWSql()
    ow.show()
    a.exec_()
    ow.settingsHandler.update_class_defaults(ow)
    ow.settingsHandler.write_defaults()
