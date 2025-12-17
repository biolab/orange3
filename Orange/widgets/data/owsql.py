from AnyQt.QtWidgets import QComboBox, QTextEdit, QMessageBox, QApplication, \
    QGridLayout, QLineEdit
from AnyQt.QtGui import QCursor
from AnyQt.QtCore import Qt

from orangewidget.utils.combobox import ComboBoxSearch

from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.data.sql.table import SqlTable, LARGE_TABLE, AUTO_DL_LIMIT
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.owbasesql import OWBaseSql
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output, Msg

MAX_DL_LIMIT = 1000000
MAX_TABLES = 1000


def is_postgres(backend):
    return getattr(backend, 'display_name', '') == "PostgreSQL"


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


class OWSql(OWBaseSql):
    name = "SQL Table"
    id = "orange.widgets.data.sql"
    description = "Load dataset from SQL."
    icon = "icons/SQLTable.svg"
    priority = 30
    category = "Data"
    keywords = "sql table, load"

    class Outputs:
        data = Output("Data", Table, doc="Attribute-valued dataset read from the input file.")

    settings_version = 2

    buttons_area_orientation = None

    TABLE, CUSTOM_SQL = range(2)
    selected_backend = Setting(None)
    data_source = Setting(TABLE)
    table = Setting(None)
    sql = Setting("")
    guess_values = Setting(True)

    materialize = Setting(False)
    materialize_table_name = Setting("")

    class Information(OWBaseSql.Information):
        data_sampled = Msg("Data description was generated from a sample.")

    class Error(OWBaseSql.Error):
        no_backends = Msg("Please install a backend to use this widget.")

    def __init__(self):
        # Lint
        self.backends = None
        self.backendcombo = None
        self.tables = None
        self.tablecombo = None
        self.tabletext = None
        self.sqltext = None
        self.custom_sql = None
        super().__init__()

    def _setup_gui(self):
        super()._setup_gui()
        self._add_backend_controls()
        self._add_tables_controls()

    def _add_backend_controls(self):
        box = self.serverbox
        self.backends = BackendModel(Backend.available_backends())
        self.backendcombo = QComboBox(box)
        if self.backends:
            self.backendcombo.setModel(self.backends)
            names = [backend.display_name for backend in self.backends]
            if self.selected_backend and self.selected_backend in names:
                self.backendcombo.setCurrentText(self.selected_backend)
        else:
            self.Error.no_backends()
            box.setEnabled(False)
        self.backendcombo.currentTextChanged.connect(self.__backend_changed)
        box.layout().insertWidget(0, self.backendcombo)

    def __backend_changed(self):
        backend = self.get_backend()
        self.selected_backend = backend.display_name if backend else None

    def _add_tables_controls(self):
        box = gui.vBox(self.controlArea, 'Data Selection')
        form = QGridLayout()
        radio_buttons = gui.radioButtons(
            box, self, 'data_source', orientation=form,
            callback=self.__on_data_source_changed)
        radio_table = gui.appendRadioButton(
            radio_buttons, 'Table:', addToLayout=False)
        radio_custom_sql = gui.appendRadioButton(
            radio_buttons, 'Custom SQL:', addToLayout=False)

        self.tables = TableModel()

        self.tablecombo = ComboBoxSearch(
            minimumContentsLength=35,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.tablecombo.setModel(self.tables)
        self.tablecombo.setToolTip('table')
        self.tablecombo.activated[int].connect(self.select_table)

        self.tabletext = QLineEdit(placeholderText='TABLE_NAME')
        self.tabletext.setToolTip('table')
        self.tabletext.editingFinished.connect(self.select_table)
        self.tabletext.setVisible(False)

        self.custom_sql = gui.vBox(box)
        self.custom_sql.setVisible(self.data_source == self.CUSTOM_SQL)
        self.sqltext = QTextEdit(self.custom_sql)
        self.sqltext.setPlainText(self.sql)
        self.custom_sql.layout().addWidget(self.sqltext)

        mt = gui.hBox(self.custom_sql)
        cb = gui.checkBox(mt, self, 'materialize', 'Materialize to table ')
        cb.setToolTip('Save results of the query in a table')
        le = gui.lineEdit(mt, self, 'materialize_table_name')
        le.setToolTip('Save results of the query in a table')

        gui.button(self.custom_sql, self, 'Execute', callback=self.open_table)

        form.addWidget(radio_table, 1, 0, Qt.AlignLeft)
        form.addWidget(self.tablecombo, 1, 1)
        form.addWidget(self.tabletext, 1, 1)
        form.addWidget(radio_custom_sql, 2, 0, Qt.AlignLeft)

        gui.checkBox(box, self, "guess_values",
                     "Auto-discover categorical variables",
                     callback=self.open_table)

    def __on_data_source_changed(self):
        self.custom_sql.setVisible(self.data_source == self.CUSTOM_SQL)
        self.select_table()

    def highlight_error(self, text=""):
        err = ['', 'QLineEdit {border: 2px solid red;}']
        self.servertext.setStyleSheet(err['server' in text or 'host' in text])
        self.usernametext.setStyleSheet(err['role' in text])
        self.databasetext.setStyleSheet(err['database' in text])

    def get_backend(self):
        if self.backendcombo.currentIndex() < 0:
            return None
        return self.backends[self.backendcombo.currentIndex()]

    def on_connection_success(self):
        super().on_connection_success()
        self.refresh_tables()
        self.select_table()

    def on_connection_error(self, err):
        super().on_connection_error(err)
        self.highlight_error(str(err).split("\n")[0])

    def clear(self):
        super().clear()
        self.highlight_error()
        self.tablecombo.clear()
        self.tablecombo.repaint()

    def refresh_tables(self):
        self.tables.clear()
        if self.backend is None:
            self.data_desc_table = None
            return

        self.tables.append("Select a table")
        if self.backend.n_tables(self.schema) <= MAX_TABLES:
            self.tables.extend(self.backend.list_tables(self.schema))
            index = self.tablecombo.findText(str(self.table))
            self.tablecombo.setCurrentIndex(index if index != -1 else 0)
            self.tablecombo.setVisible(True)
            self.tabletext.setVisible(False)
        else:
            self.tablecombo.setVisible(False)
            self.tabletext.setVisible(True)
        self.tablecombo.repaint()

    # Called on tablecombo selection change:
    def select_table(self):
        if self.data_source == self.TABLE:
            return self.open_table()
        else:
            self.data_desc_table = None
            if self.database_desc:
                self.database_desc["Table"] = "(None)"
            self.table = None
            if len(str(self.sql)) > 14:
                return self.open_table()
        return None

    def get_table(self):
        if self.backend is None:
            return None
        curIdx = self.tablecombo.currentIndex()
        if self.data_source == self.TABLE and curIdx <= 0 and \
                self.tabletext.text() == "":
            if self.database_desc:
                self.database_desc["Table"] = "(None)"
            self.data_desc_table = None
            return None

        if self.data_source == self.TABLE:
            self.table = self.tables[curIdx] if curIdx > 0 else \
                self.tabletext.text()
            self.database_desc["Table"] = self.table
            if "Query" in self.database_desc:
                del self.database_desc["Query"]
            what = self.table
        else:
            what = self.sql = self.sqltext.toPlainText()
            self.table = "Custom SQL"
            if self.materialize:
                if not self.materialize_table_name:
                    self.Error.connection(
                        "Specify a table name to materialize the query")
                    return None
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
                except BackendError as ex:
                    self.Error.connection(str(ex))
                    return None

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
            return None

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
            if is_postgres(self.backend):
                sample_button = confirm.addButton("Yes, on a sample",
                                                  QMessageBox.YesRole)
            confirm.exec()
            if confirm.clickedButton() == no_button:
                self.guess_values = False
            elif is_postgres(self.backend) and \
                    confirm.clickedButton() == sample_button:
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

        if table.approx_len() > AUTO_DL_LIMIT:
            if is_postgres(self.backend):
                confirm = QMessageBox(self)
                confirm.setIcon(QMessageBox.Warning)
                confirm.setText("Data appears to be big. Do you really "
                                "want to download it to local memory?\n"
                                "Table length: {:,}. Limit {:,}".format(
                    table.approx_len(), MAX_DL_LIMIT))

                if table.approx_len() <= MAX_DL_LIMIT:
                    confirm.addButton("Yes", QMessageBox.YesRole)
                no_button = confirm.addButton("No", QMessageBox.NoRole)
                sample_button = confirm.addButton("Yes, a sample",
                                                  QMessageBox.YesRole)
                confirm.exec()
                if confirm.clickedButton() == no_button:
                    return None
                elif confirm.clickedButton() == sample_button:
                    table = table.sample_percentage(
                        AUTO_DL_LIMIT / table.approx_len() * 100)
            else:
                if table.approx_len() > MAX_DL_LIMIT:
                    QMessageBox.warning(
                        self, 'Warning',
                        "Data is too big to download.\n"
                        "Table length: {:,}. Limit {:,}".format(table.approx_len(), MAX_DL_LIMIT)
                    )
                    return None
                else:
                    confirm = QMessageBox.question(
                        self, 'Question',
                        "Data appears to be big. Do you really "
                        "want to download it to local memory?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if confirm == QMessageBox.No:
                        return None

        table.download_data(MAX_DL_LIMIT)
        table = Table(table)

        return table

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            # Until Orange version 3.4.4 username and password had been stored
            # in Settings.
            cm = cls._credential_manager(settings["host"], settings["port"])
            cm.username = settings["username"]
            cm.password = settings["password"]


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSql).run()
