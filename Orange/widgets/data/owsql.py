import sys
from collections import OrderedDict

from AnyQt.QtWidgets import (
    QLineEdit, QComboBox, QTextEdit, QMessageBox, QSizePolicy, QApplication, QDialog, QPushButton, QVBoxLayout,
    QDialogButtonBox, QErrorMessage)
from AnyQt.QtGui import QCursor
from AnyQt.QtCore import Qt, QTimer, QSize, pyqtSignal

from Orange.canvas import report
from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.data.sql.table import SqlTable, LARGE_TABLE, AUTO_DL_LIMIT
from Orange.widgets import gui
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.widget import OWWidget, OutputSignal, Msg

MAX_DL_LIMIT = 1000000


class TableModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return str(self[row])
        return super().data(index, role)


class MyCombo(QComboBox):
    def currentItem(self):
        if len(self.model()):
            return self.model()[self.currentIndex()]


class OWSql(OWWidget):
    name = "SQL Table"
    id = "orange.widgets.data.sql"
    description = "Load data set from SQL."
    icon = "icons/SQLTable.svg"
    priority = 10
    category = "Data"
    keywords = ["data", "file", "load", "read"]
    outputs = [OutputSignal(
        "Data", Table,
        doc="Attribute-valued data set read from the input file.")]

    want_main_area = False

    _connections = None

    class ConnectionsSetting(Setting):
        def __get__(self, instance, type=None):
            if instance is None:
                return self
            return [c.encode() for c in instance.connections_combo.model()]

        def __set__(self, instance, value):
            instance._connections = [Connection.decode(c) for c in value]

    connections = ConnectionsSetting([])

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

        vbox = gui.vBox(self.controlArea, "Connection", addSpace=True)
        box = gui.vBox(vbox)

        hbox = gui.hBox(box)
        model = TableModel(self._connections)
        self.connections_combo = MyCombo()
        self.connections_combo.setModel(model)
        self.connections_combo.currentIndexChanged[int].connect(self.connect)
        hbox.layout().addWidget(self.connections_combo)

        dlg = EditConnection(self)
        dlg.connected.connect(self.update_connection)
        dlg.deleted.connect(self.remove_connection)
        self.edit_button = btn = QPushButton("✎", hbox)
        btn.clicked.connect(lambda x:
            dlg.show(self.connections_combo.currentItem()))
        btn.setEnabled(len(self.connections_combo.model()) > 0)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hbox.layout().addWidget(btn)

        btn = QPushButton("+", hbox)
        btn.clicked.connect(lambda x: dlg.show())
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hbox.layout().addWidget(btn)

        tables = gui.hBox(box)
        self.tablemodel = TableModel()
        self.tablecombo = MyCombo(
            minimumContentsLength=35,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength
        )
        self.tablecombo.setModel(self.tablemodel)
        self.tablecombo.setToolTip('table')
        tables.layout().addWidget(self.tablecombo)
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
                     "Auto-discover discrete variables",
                     callback=self.open_table)

        gui.checkBox(box, self, "download",
                     "Download data to local memory",
                     callback=self.open_table)

        gui.rubber(self.buttonsArea)
        QTimer.singleShot(0, self.connect)

    def update_connection(self, connection):
        if connection not in self.connections_combo.model():
            self.connections_combo.model().insert(0, connection)
            self.connections_combo.setCurrentIndex(0)
        self.edit_button.setEnabled(len(self.connections_combo.model()) > 0)

    def remove_connection(self, connection):
        if connection in self.connections_combo.model():
            self.connections_combo.model().remove(connection)
        self.edit_button.setEnabled(len(self.connections_combo.model()) > 0)

    def connect(self):
        connection = self.connections_combo.currentItem()
        if not connection:
            return

        try:
            self.backend = connection.connect()
            self.Error.connection.clear()
            self.refresh_tables()
            self.select_table()
        except BackendError as err:
            error = str(err).split('\n')[0]
            self.Error.connection(error)
            self.tablecombo.clear()

    def refresh_tables(self):
        self.tablemodel.clear()
        self.Error.missing_extension.clear()
        if self.backend is None:
            return

        self.tablemodel.append("Select a table")
        self.tablemodel.extend(self.backend.list_tables(self.connections_combo.currentItem().schema))
        self.tablemodel.append("Custom SQL")

    def select_table(self):
        curIdx = self.tablecombo.currentIndex()
        if self.tablecombo.itemText(curIdx) != "Custom SQL":
            self.custom_sql.setVisible(False)
            return self.open_table()
        else:
            self.custom_sql.setVisible(True)
            self.table = None

        #self.Error.missing_extension(
        #    's' if len(missing) > 1 else '',
        #    ', '.join(missing),
        #    shown=missing)

    def open_table(self):
        table = self.get_table()
        self.send("Data", table)

    def get_table(self):
        if self.tablecombo.currentIndex() <= 0:
            return

        if self.tablecombo.currentIndex() < self.tablecombo.count() - 1:
            self.table = self.tablemodel[self.tablecombo.currentIndex()]
        else:
            self.sql = self.table = self.sqltext.toPlainText()
            if self.materialize:
                import psycopg2
                if not self.materialize_table_name:
                    self.Error.connection(
                        "Specify a table name to materialize the query")
                    return
                try:
                    with self.backend.execute_sql_query("DROP TABLE IF EXISTS " + self.materialize_table_name):
                        pass
                    with self.backend.execute_sql_query("CREATE TABLE " + self.materialize_table_name + " AS " + self.table):
                        pass
                    with self.backend.execute_sql_query("ANALYZE " + self.materialize_table_name):
                        pass
                    self.table = self.materialize_table_name
                except psycopg2.ProgrammingError as ex:
                    self.Error.connection(str(ex))
                    return

        try:
            connection = self.connections_combo.currentItem()
            table = SqlTable(connection.parameters,
                             self.table,
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
        connection = self.connections_combo.currentItem()
        if not connection:
            self.report_paragraph("No database connection.")
            return
        parameters = connection.parameters
        parameters.pop("password", None)
        self.report_items([("connection", str(connection))])


class Connection:
    def __init__(self):
        self.backend = ""
        self.host = ""
        self.port = ""
        self.database = ""
        self.schema = ""
        self.username = ""
        self._password = ""

    def connect(self):
        for backend in Backend.available_backends():
            if backend.display_name == self.backend:
                return backend(self.parameters)
        raise BackendError("Backend {} is not installed".format(self.backend))

    @property
    def parameters(self):
        params = OrderedDict()
        if self.host:
            params["host"] = self.host
        if self.port:
            params["port"] = self.port
        if self.database:
            params["database"] = self.database
        if self.schema:
            params["schema"] = self.schema
        if self.username:
            params["user"] = self.username
        if self.password:
            params["password"] = self.password
        return params

    @property
    def server_port(self):
        if self.host:
            if self.port:
                return '{}:{}'.format(self.host, self.port)
            else:
                return self.host
        else:
            return ""

    @server_port.setter
    def server_port(self, value):
        self.host, _, self.port = value.partition(":")

    @property
    def database_schema(self):
        if self.database:
            if self.schema:
                return '{}/{}'.format(self.database, self.schema)
            else:
                return self.database
        else:
            return ""

    @database_schema.setter
    def database_schema(self, value):
        self.database, _, self.schema = value.partition("/")

    @property
    def password(self):
        if not self._password:
            cm = CredentialManager("/".join([self.host, self.username]))
            self._password = cm.password
        return self._password

    @password.setter
    def password(self, value):
        cm = CredentialManager("/".join([self.host, self.username]))
        self._password = cm.password = value

    def encode(self):
        c = dict(self.__dict__)
        c.pop("_password")
        return c

    @staticmethod
    def decode(parameters):
        c = Connection()
        c.__dict__.update(parameters)
        return c

    def __str__(self):
        user = "{}@".format(self.username) if self.username else ""
        database = ", db={}".format(self.database) if self.database else ""
        return "{}{}{}".format(user, self.host, database)


class BackendModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return self[row].display_name
        return super().data(index, role)


class EditConnection(QDialog):
    connection = None

    connected = pyqtSignal('PyQt_PyObject')
    deleted = pyqtSignal('PyQt_PyObject')

    def __init__(self, parent):
        super().__init__(parent, Qt.Dialog)

        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.backend = self.create_backend_combo()
        self.server = self.create_line_edit('Server')
        self.database = self.create_line_edit(
            'Database[/Schema]', 'Database or optionally Database/Schema')
        self.username = self.create_line_edit('Username')
        self.password = self.create_line_edit('Password')
        self.password.setEchoMode(QLineEdit.Password)

        self.layout().addStretch(1)

        btns = QDialogButtonBox(self)
        ok = QPushButton("Connect")
        ok.setEnabled(False)
        ok.clicked.connect(self.save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(lambda x: self.hide())
        self.deletebtn = delete = QPushButton("Delete")
        delete.clicked.connect(self.delete)

        btns.addButton(ok, QDialogButtonBox.AcceptRole)
        btns.addButton(cancel, QDialogButtonBox.RejectRole)
        btns.addButton(delete, QDialogButtonBox.ResetRole)

        self.layout().addWidget(btns)

        # ok button is enabled when server name is set
        self.server.textChanged.connect(
            lambda x: ok.setEnabled(self.server.text() != ""))

    def sizeHint(self):
        return QSize(500, 250)

    def create_backend_combo(self):
        model = BackendModel(Backend.available_backends())
        combo = MyCombo(self)
        if len(model):
            combo.setModel(model)
        else:
            self.setEnabled(False)
        self.layout().addWidget(combo)
        return combo

    def create_line_edit(self, placeholder, tooltip=None):
        tooltip = tooltip or placeholder

        lineedit = QLineEdit(self)
        lineedit.setPlaceholderText(placeholder)
        lineedit.setToolTip(tooltip)
        self.layout().addWidget(lineedit)
        return lineedit

    def show(self, connection=None):
        new = connection is None
        connection = connection or Connection()

        if new:
            self.setWindowTitle("New connection")
            self.deletebtn.hide()
        else:
            self.setWindowTitle("Edit connection")
            self.deletebtn.show()

        self.connection = connection
        if connection.backend:
            selected_backend = [b for b in Backend.available_backends()
                                if b.display_name == connection.backend]
            if selected_backend:
                self.backend.setCurrentIndex(
                    self.backend.model().indexOf(selected_backend[0]))
        self.server.setText(connection.server_port)
        self.database.setText(connection.database_schema)
        self.username.setText(connection.username)
        self.password.setText(connection.password)
        super().show()

    def save(self):
        backend = self.backend.model()[self.backend.currentIndex()]
        self.connection.backend = backend.display_name
        self.connection.server_port = self.server.text()
        self.connection.database_schema = self.database.text()
        self.connection.username = self.username.text()
        self.connection.password = self.password.text()

        try:
            self.connection.connect()
            self.connected.emit(self.connection)
            self.hide()
        except BackendError as err:
            message = QMessageBox(self)
            message.setIcon(QMessageBox.Critical)
            message.setText("Could not connect to {}\n\n"
                            "Additional information:\n"
                            "{}".format(
                                self.connection.host, str(err)))
            message.show()

    def delete(self):
        print("deleting")
        self.deleted.emit(self.connection)
        self.hide()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWSql()
    ow.show()
    a.exec_()
    ow.saveSettings()
