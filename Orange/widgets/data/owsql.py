import sys

import psycopg2
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QCursor, QMessageBox

from Orange.data import Table
from Orange.data.sql.table import SqlTable, LARGE_TABLE
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWSql(widget.OWWidget):
    name = "SQL Table"
    id = "orange.widgets.data.sql"
    description = """
    Load dataset from SQL."""
    long_description = """
    Sql widget connects to server and opens data from there. """
    icon = "icons/SQLTable.svg"
    author = "Anze Staric"
    maintainer_email = "anze.staric@fri.uni-lj.si"
    priority = 10
    category = "Data"
    keywords = ["data", "file", "load", "read"]
    outputs = [{"name": "Data",
                "type": Table,
                "doc": "Attribute-valued data set read from the input file."}]

    want_main_area = False

    host = Setting(None)
    port = Setting(None)
    database = Setting(None)
    username = Setting(None)
    password = Setting(None)
    table = Setting(None)
    sql = Setting("")
    guess_values = Setting(True)

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
            self.servertext.setText(self.host if not self.port else
                                    '{}:{}'.format(self.host, self.port))
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

        tables.layout().addWidget(self.tablecombo)
        self.tablecombo.activated[int].connect(self.select_table)
        self.connectbutton = gui.button(
            tables, self, '↻', callback=self.connect)
        self.connectbutton.setSizePolicy(
            QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        tables.layout().addWidget(self.connectbutton)

        self.custom_sql = gui.widgetBox(box, orientation='vertical')
        self.custom_sql.setVisible(False)
        self.sqltext = QtGui.QTextEdit(self.custom_sql)
        self.sqltext.setPlainText(self.sql)
        self.custom_sql.layout().addWidget(self.sqltext)

        self.executebtn = gui.button(
            self.custom_sql, self, 'Execute', callback=self.open_table)

        box.layout().addWidget(self.custom_sql)

        gui.checkBox(box, self, "guess_values",
                     "Auto-discover discrete variables.",
                     callback=self.open_table)
        self.connect()
        if self.table:
            self.open_table()

    def error(self, id=0, text=""):
        super().error(id, text)
        if 'server' in text or 'host' in text:
            self.servertext.setStyleSheet('QLineEdit {border: 2px solid red;}')
        else:
            self.servertext.setStyleSheet('')
        if 'role' in text:
            self.usernametext.setStyleSheet('QLineEdit {border: 2px solid red;}')
        else:
            self.usernametext.setStyleSheet('')
        if 'database' in text:
            self.databasetext.setStyleSheet('QLineEdit {border: 2px solid red;}')
        else:
            self.databasetext.setStyleSheet('')


    def connect(self):
        hostport = self.servertext.text().split(':')
        self.host = hostport[0]
        self.port = hostport[1] if len(hostport) == 2 else None
        self.database = self.databasetext.text()
        self.username = self.usernametext.text() or None
        self.password = self.passwordtext.text() or None
        try:
            self._connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            self.error(0)
            self.refresh_tables()
        except psycopg2.Error as err:
            self.error(0, str(err).split('\n')[0])
            self.tablecombo.clear()


    def refresh_tables(self):
        self.tablecombo.clear()
        if self._connection is None:
            return

        cur = self._connection.cursor()
        cur.execute("""SELECT --n.nspname as "Schema",
                              c.relname AS "Name"
                       FROM pg_catalog.pg_class c
                  LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                      WHERE c.relkind IN ('r','v','m','S','f','')
                        AND n.nspname <> 'pg_catalog'
                        AND n.nspname <> 'information_schema'
                        AND n.nspname !~ '^pg_toast'
                        AND pg_catalog.pg_table_is_visible(c.oid)
                   ORDER BY 1;""")

        self.tablecombo.addItem("Select a table")
        for i, (table_name,) in enumerate(cur.fetchall()):
            self.tablecombo.addItem(table_name)
            if table_name == self.table:
                self.tablecombo.setCurrentIndex(i + 1)
        self.tablecombo.addItem("Custom SQL")

    def select_table(self):
        curIdx = self.tablecombo.currentIndex()
        if self.tablecombo.itemText(curIdx) != "Custom SQL":
            self.custom_sql.setVisible(False)
            return self.open_table()
        else:
            self.custom_sql.setVisible(True)
            self.table = None

    def open_table(self):
        if self.tablecombo.currentIndex() <= 0:
            return

        if self.tablecombo.currentIndex() < self.tablecombo.count() - 1:
            self.table = self.tablecombo.currentText()
        else:
            self.table = self.sqltext.toPlainText()

        table = SqlTable(dict(host=self.host,
                              port=self.port,
                              database=self.database,
                              user=self.username,
                              password=self.password),
                         self.table,
                         inspect_values=False)
        sample = False
        if table.approx_len() > LARGE_TABLE and self.guess_values:
            confirm = QMessageBox(self)
            confirm.setIcon(QMessageBox.Warning)
            confirm.setText("Attribute discovery might take "
                            "a long time on large tables.\n"
                            "Do you want to auto discover attributes?")
            confirm.addButton("Yes", QMessageBox.YesRole)
            no_button = confirm.addButton("No", QMessageBox.NoRole)
            sample_button = confirm.addButton("Yes, on a sample",
                                              QMessageBox.YesRole)
            confirm.exec()
            if confirm.clickedButton() == no_button:
                self.guess_values = False
            elif confirm.clickedButton() == sample_button:
                sample = True

        if self.guess_values:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            if sample:
                s = table.sample_time(1)
                domain = s.get_domain(guess_values=True)
                self.information(
                    1, "Domain was generated from a sample of the table.")
            else:
                domain = table.get_domain(guess_values=True)
            QApplication.restoreOverrideCursor()
            table.domain = domain

        self.send("Data", table)

if __name__ == "__main__":
    import os

    a = QtGui.QApplication(sys.argv)
    settings = os.path.join(widget.environ.widget_settings_dir,
                            OWSql.name + ".ini")
    ow = OWSql()
    ow.show()
    a.exec_()
    ow.saveSettings()
