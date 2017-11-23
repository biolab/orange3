# -*- coding: utf-8 -*-
import sys
from collections import OrderedDict

from AnyQt.QtWidgets import (
    QLineEdit, QComboBox, QTextEdit, QMessageBox, QSizePolicy, QApplication)
from AnyQt.QtGui import QCursor
from AnyQt.QtCore import Qt, QTimer

from Orange.widgets import report
from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.data.sql.table import SqlTable, LARGE_TABLE, AUTO_DL_LIMIT
from Orange.widgets import gui
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.widget import OWWidget, Output, Msg

MAX_DL_LIMIT = 1000000


class TableModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return str(self[row])
        return super().data(index, role)


class BackendModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return self[row].display_name
        return super().data(index, role)


class OWSql(OWWidget):
    name = "SQL Table"
    id = "orange.widgets.data.sql"
    description = "Load dataset from SQL."
    icon = "icons/SQLTable.svg"
    priority = 30
    category = "Data"
    keywords = ["data", "file", "load", "read", "SQL"]

    class Outputs:
        data = Output("Data", Table, doc="Attribute-valued dataset read from the input file.")

    settings_version = 2

    want_main_area = False
    resizing_enabled = False

    host = Setting(None)
    port = Setting(None)
    database = Setting(None)
    schema = Setting(None)
    username = ""
    password = ""
    table = Setting(None)
    sql = Setting("")
    guess_values = Setting(True)
    download = Setting(False)

    materialize = Setting(False)
    materialize_table_name = Setting("")

    class Information(OWWidget.Information):
        data_sampled = Msg("Data description was generated from a sample.")

    class Error(OWWidget.Error):
        connection = Msg("{}")
        no_backends = Msg("Please install a backend to use this widget")
        missing_extension = Msg("Database is missing extension{}: {}")

    def __init__(self):
        super().__init__()

        self.backend = None
        self.data_desc_table = None
        self.database_desc = None

        vbox = gui.vBox(self.controlArea, "Server", addSpace=True)
        box = gui.vBox(vbox)

        self.backends = BackendModel(Backend.available_backends())
        self.backendcombo = QComboBox(box)
        if len(self.backends):
            self.backendcombo.setModel(self.backends)
        else:
            self.Error.no_backends()
            box.setEnabled(False)
        box.layout().addWidget(self.backendcombo)

        self.servertext = QLineEdit(box)
        self.servertext.setPlaceholderText('Server')
        self.servertext.setToolTip('Server')
        self.servertext.editingFinished.connect(self._load_credentials)
        if self.host:
            self.servertext.setText(self.host if not self.port else
                                    '{}:{}'.format(self.host, self.port))
        box.layout().addWidget(self.servertext)

        self.databasetext = QLineEdit(box)
        self.databasetext.setPlaceholderText('Database[/Schema]')
        self.databasetext.setToolTip('Database or optionally Database/Schema')
        if self.database:
            self.databasetext.setText(
                self.database if not self.schema else
                '{}/{}'.format(self.database, self.schema))
        box.layout().addWidget(self.databasetext)
        self.usernametext = QLineEdit(box)
        self.usernametext.setPlaceholderText('Username')
        self.usernametext.setToolTip('Username')

        box.layout().addWidget(self.usernametext)
        self.passwordtext = QLineEdit(box)
        self.passwordtext.setPlaceholderText('Password')
        self.passwordtext.setToolTip('Password')
        self.passwordtext.setEchoMode(QLineEdit.Password)

        box.layout().addWidget(self.passwordtext)

        self._load_credentials()
        self.tables = TableModel()

        tables = gui.hBox(box)
        self.tablecombo = QComboBox(
            minimumContentsLength=35,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength
        )
        self.tablecombo.setModel(self.tables)
        self.tablecombo.setToolTip('table')
        tables.layout().addWidget(self.tablecombo)
        self.connect()

        index = self.tablecombo.findText(str(self.table))
        if index != -1:
            self.tablecombo.setCurrentIndex(index)
        # set up the callback to select_table in case of selection change
        self.tablecombo.activated[int].connect(self.select_table)

        self.connectbutton = gui.button(
            tables, self, '↻', callback=self.connect)
        self.connectbutton.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed)
        tables.layout().addWidget(self.connectbutton)

        self.custom_sql = gui.vBox(box)
        self.custom_sql.setVisible(False)
        self.sqltext = QTextEdit(self.custom_sql)
        self.sqltext.setPlainText(self.sql)
        self.custom_sql.layout().addWidget(self.sqltext)

        mt = gui.hBox(self.custom_sql)
        cb = gui.checkBox(mt, self, 'materialize', 'Materialize to table ')
        cb.setToolTip('Save results of the query in a table')
        le = gui.lineEdit(mt, self, 'materialize_table_name')
        le.setToolTip('Save results of the query in a table')

        self.executebtn = gui.button(
            self.custom_sql, self, 'Execute', callback=self.open_table)

        box.layout().addWidget(self.custom_sql)

        gui.checkBox(box, self, "guess_values",
                     "Auto-discover categorical variables",
                     callback=self.open_table)

        gui.checkBox(box, self, "download",
                     "Download data to local memory",
                     callback=self.open_table)

        gui.rubber(self.buttonsArea)

        QTimer.singleShot(0, self.select_table)

    def _load_credentials(self):
        self._parse_host_port()
        cm = self._credential_manager(self.host, self.port)
        self.username = cm.username
        self.password = cm.password

        if self.username:
            self.usernametext.setText(self.username)
        if self.password:
            self.passwordtext.setText(self.password)

    def _save_credentials(self):
        cm = self._credential_manager(self.host, self.port)
        cm.username = self.username or ''
        cm.password = self.password or ''

    def _credential_manager(self, host, port):
        return CredentialManager("SQL Table: {}:{}".format(host, port))

    def error(self, id=0, text=""):
        super().error(id, text)
        err_style = 'QLineEdit {border: 2px solid red;}'
        if 'server' in text or 'host' in text:
            self.servertext.setStyleSheet(err_style)
        else:
            self.servertext.setStyleSheet('')
        if 'role' in text:
            self.usernametext.setStyleSheet(err_style)
        else:
            self.usernametext.setStyleSheet('')
        if 'database' in text:
            self.databasetext.setStyleSheet(err_style)
        else:
            self.databasetext.setStyleSheet('')

    def _parse_host_port(self):
        hostport = self.servertext.text().split(':')
        self.host = hostport[0]
        self.port = hostport[1] if len(hostport) == 2 else None

    def connect(self):
        self._parse_host_port()
        self.database, _, self.schema = self.databasetext.text().partition('/')
        self.username = self.usernametext.text() or None
        self.password = self.passwordtext.text() or None
        try:
            if self.backendcombo.currentIndex() < 0:
                return
            backend = self.backends[self.backendcombo.currentIndex()]
            self.backend = backend(dict(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            ))
            self.Error.connection.clear()
            self._save_credentials()
            self.database_desc = OrderedDict((
                ("Host", self.host), ("Port", self.port),
                ("Database", self.database), ("User name", self.username)
            ))
            self.refresh_tables()
        except BackendError as err:
            error = str(err).split('\n')[0]
            self.Error.connection(error)
            self.database_desc = self.data_desc_table = None
            self.tablecombo.clear()

    def refresh_tables(self):
        self.tables.clear()
        self.Error.missing_extension.clear()
        if self.backend is None:
            self.data_desc_table = None
            return

        self.tables.append("Select a table")
        self.tables.append("Custom SQL")
        self.tables.extend(self.backend.list_tables(self.schema))

    # Called on tablecombo selection change:
    def select_table(self):
        curIdx = self.tablecombo.currentIndex()
        if self.tablecombo.itemText(curIdx) != "Custom SQL":
            self.custom_sql.setVisible(False)
            return self.open_table()
        else:
            self.custom_sql.setVisible(True)
            self.data_desc_table = None
            self.database_desc["Table"] = "(None)"
            self.table = None
            if len(str(self.sql)) > 14:
                return self.open_table()

        #self.Error.missing_extension(
        #    's' if len(missing) > 1 else '',
        #    ', '.join(missing),
        #    shown=missing)

    def open_table(self):
        table = self.get_table()
        self.data_desc_table = table
        self.Outputs.data.send(table)

    def get_table(self):
        curIdx = self.tablecombo.currentIndex()
        if curIdx <= 0:
            if self.database_desc:
                self.database_desc["Table"] = "(None)"
            self.data_desc_table = None
            return

        if self.tablecombo.itemText(curIdx) != "Custom SQL":
            self.table = self.tables[self.tablecombo.currentIndex()]
            self.database_desc["Table"] = self.table
            if "Query" in self.database_desc:
                del self.database_desc["Query"]
            what = self.table
        else:
            what = self.sql = self.sqltext.toPlainText()
            self.table = "Custom SQL"
            if self.materialize:
                import psycopg2
                if not self.materialize_table_name:
                    self.Error.connection(
                        "Specify a table name to materialize the query")
                    return
                try:
                    with self.backend.execute_sql_query("DROP TABLE IF EXISTS " +
                                                        self.materialize_table_name):
                        pass
                    with self.backend.execute_sql_query("CREATE TABLE " +
                                                        self.materialize_table_name +
                                                        " AS " + self.sql):
                        pass
                    with self.backend.execute_sql_query("ANALYZE " + self.materialize_table_name):
                        pass
                except (psycopg2.ProgrammingError, BackendError) as ex:
                    self.Error.connection(str(ex))
                    return

        try:
            table = SqlTable(dict(host=self.host,
                                  port=self.port,
                                  database=self.database,
                                  user=self.username,
                                  password=self.password),
                             what,
                             backend=type(self.backend),
                             inspect_values=False)
        except BackendError as ex:
            self.Error.connection(str(ex))
            return

        self.Error.connection.clear()

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

        self.Information.clear()
        if self.guess_values:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            if sample:
                s = table.sample_time(1)
                domain = s.get_domain(inspect_values=True)
                self.Information.data_sampled()
            else:
                domain = table.get_domain(inspect_values=True)
            QApplication.restoreOverrideCursor()
            table.domain = domain

        if self.download:
            if table.approx_len() > MAX_DL_LIMIT:
                QMessageBox.warning(
                    self, 'Warning', "Data is too big to download.\n"
                    "Consider using the Data Sampler widget to download "
                    "a sample instead.")
                self.download = False
            elif table.approx_len() > AUTO_DL_LIMIT:
                confirm = QMessageBox.question(
                    self, 'Question', "Data appears to be big. Do you really "
                                      "want to download it to local memory?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if confirm == QMessageBox.No:
                    self.download = False
        if self.download:
            table.download_data(MAX_DL_LIMIT)
            table = Table(table)

        return table

    def send_report(self):
        if not self.database_desc:
            self.report_paragraph("No database connection.")
            return
        self.report_items("Database", self.database_desc)
        if self.data_desc_table:
            self.report_items("Data",
                              report.describe_data(self.data_desc_table))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            # Until Orange version 3.4.4 username and password had been stored
            # in Settings.
            cm = cls._credential_manager(settings["host"], settings["port"])
            cm.username = settings["username"]
            cm.password = settings["password"]


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWSql()
    ow.show()
    a.exec_()
    ow.saveSettings()
