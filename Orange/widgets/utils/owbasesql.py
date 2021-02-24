import os
from typing import Type
from collections import OrderedDict

from AnyQt.QtWidgets import QLineEdit, QSizePolicy

from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.alchemy_base import SqliteAlchemy
from Orange.data.sql.backend.base import BackendError
from Orange.widgets import gui, report
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import OWWidget, Msg


# the name of the field (placeholder and the tooltip)
GUI_FIELDS = {
    "servertext": ("Server", "Server"),
    "databasetext": ("Database[/Schema]", "Database or optionally Database/Schema"),
    "usernametext": ("Username", "Username"),
    "passwordtext": ("Password", "Password")
}


class OWBaseSql(OWWidget, openclass=True):
    """Base widget for connecting to a database.
    Override `get_backend` when subclassing to get corresponding backend.
    """
    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        connection = Msg("{}")

    want_main_area = False
    resizing_enabled = False

    host = Setting(None)  # type: Optional[str]
    port = Setting(None)  # type: Optional[str]
    database = Setting(None)  # type: Optional[str]
    schema = Setting(None)  # type: Optional[str]
    username = ""
    password = ""

    def __init__(self):
        super().__init__()
        self.backend = None  # type: Optional[Backend]
        self.data_desc_table = None  # type: Optional[Table]
        self.database_desc = None  # type: Optional[OrderedDict]
        self._setup_gui()
        self.connect()

    def _setup_gui(self):
        self.controlArea.setMinimumWidth(360)

        vbox = gui.vBox(self.controlArea, "Server", addSpace=True)
        self.serverbox = gui.vBox(vbox)
        self.servertext = QLineEdit(self.serverbox)
        self.servertext.setPlaceholderText(GUI_FIELDS["servertext"][0])
        self.servertext.setToolTip(GUI_FIELDS["servertext"][1])
        self.servertext.editingFinished.connect(self._load_credentials)
        if self.host:
            self.servertext.setText(self.host if not self.port else
                                    "{}:{}".format(self.host, self.port))
        self.serverbox.layout().addWidget(self.servertext)

        self.databasetext = QLineEdit(self.serverbox)
        self.databasetext.setPlaceholderText(GUI_FIELDS["databasetext"][0])
        self.databasetext.setToolTip(GUI_FIELDS["databasetext"][1])
        if self.database:
            self.databasetext.setText(
                self.database if not self.schema else
                "{}/{}".format(self.database, self.schema))
        self.serverbox.layout().addWidget(self.databasetext)
        self.usernametext = QLineEdit(self.serverbox)
        self.usernametext.setPlaceholderText(GUI_FIELDS["usernametext"][0])
        self.usernametext.setToolTip(GUI_FIELDS["usernametext"][1])

        self.serverbox.layout().addWidget(self.usernametext)
        self.passwordtext = QLineEdit(self.serverbox)
        self.passwordtext.setPlaceholderText(GUI_FIELDS["passwordtext"][0])
        self.passwordtext.setToolTip(GUI_FIELDS["passwordtext"][1])
        self.passwordtext.setEchoMode(QLineEdit.Password)

        self.serverbox.layout().addWidget(self.passwordtext)

        self._load_credentials()

        self.connectbutton = gui.button(self.serverbox, self, "Connect",
                                        callback=self.connect)
        self.connectbutton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

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
        cm.username = self.username or ""
        cm.password = self.password or ""

    @staticmethod
    def _credential_manager(host, port):
        return CredentialManager("SQL Table: {}:{}".format(host, port))

    def _parse_host_port(self):
        hostport = self.servertext.text().split(":")
        self.host = hostport[0]
        self.port = hostport[1] if len(hostport) == 2 else None

    def _check_db_settings(self):
        self._parse_host_port()
        if self.get_backend() == SqliteAlchemy:
            self.database = self.databasetext.text()
            if os.path.isfile(self.database):
                # no schema provided together with the name
                self.schema = ""
            else:
                # schema as a last part of the name
                self.database, _, self.schema = self.database.partition("/")
        else:
            self.database, _, self.schema = self.databasetext.text().partition("/")
        self.username = self.usernametext.text() or None
        self.password = self.passwordtext.text() or None

    def connect(self):
        self.clear()
        self._check_db_settings()
        if not self.database:
            return
        try:
            backend = self.get_backend()
            if backend is None:
                return
            self.backend = backend(dict(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            ))
            self.on_connection_success()
        except BackendError as err:
            self.on_connection_error(err)

    def get_backend(self) -> Type[Backend]:
        """
        Derived widgets should override this to get corresponding backend.

        Returns
        -------
        backend: Type[Backend]
        """
        raise NotImplementedError

    def on_connection_success(self):
        self._save_credentials()
        self.database_desc = OrderedDict((
            ("Host", self.host), ("Port", self.port),
            ("Database", self.database), ("User name", self.username)
        ))

    def on_connection_error(self, err):
        error = str(err).split("\n")[0]
        self.Error.connection(error)

    def open_table(self):
        data = self.get_table()
        self.data_desc_table = data
        self.Outputs.data.send(data)
        info = data.approx_len() if data else self.info.NoOutput
        detail = format_summary_details(data) if data else ""
        self.info.set_output_summary(info, detail)

    def get_table(self) -> Table:
        """
        Derived widgets should override this to get corresponding table.

        Returns
        -------
        table: Table
        """
        raise NotImplementedError

    def clear(self):
        self.Error.connection.clear()
        self.database_desc = None
        self.data_desc_table = None
        self.Outputs.data.send(None)
        self.info.set_output_summary(self.info.NoOutput)

    def send_report(self):
        if not self.database_desc:
            self.report_paragraph("No database connection.")
            return
        self.report_items("Database", self.database_desc)
        if self.data_desc_table:
            self.report_items("Data",
                              report.describe_data(self.data_desc_table))
