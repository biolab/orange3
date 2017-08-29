from AnyQt.QtWidgets import QSizePolicy, QPlainTextEdit, QHBoxLayout, QLineEdit
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import OWWidget
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import pandas
import sip
import numpy as np
import cx_Oracle

class ORACLESQL(OWWidget):

    name = "Oracle SQL"
    icon = "icons/ORACLE_SQL.svg"
    want_main_area = False
    inputs = []
    outputs = [("Dataframe", pandas.DataFrame, widget.Default),
               ("Data", Table, widget.Default)]
    description = "Create a Table from an ODBC datasource"
    settingsHandler = settings.DomainContextHandler()
    priority = 1
    autocommit = settings.Setting(False, schema_only=True)
    savedQuery = settings.Setting(None, schema_only=True)
    savedUsername = settings.Setting(None, schema_only=True)
    savedPwd = settings.Setting(None, schema_only=True)
    savedDB = settings.Setting(None, schema_only=True)

    def __init__(self):
        super().__init__()

        #label = QtGui.QLabel("connection with oracle database")
        #Defaults
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.domain = None
        self.data = None
        self.query = ''
        if self.savedQuery is not None:
            self.query = self.savedQuery
        self.username = ''
        if self.savedUsername is not None:
            self.username = self.savedUsername
        self.password = ''
        if self.savedPwd is not None:
            self.password = self.savedPwd
        self.database = ''
        if self.savedDB is not None:
            self.database = self.savedDB
        #Control Area layout
        sip.delete(self.controlArea.layout())
        self.controlArea.setLayout(QHBoxLayout())
        self.connectBox = gui.widgetBox(self.controlArea, "Database connection")
        self.connectBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.sqlBox = gui.widgetBox(self.controlArea, "SQL")
        #Database
        self.userLabel = gui.label(self.connectBox, self, 'User name')
        self.connectUser = QLineEdit(self.username, self)
        self.connectBox.layout().addWidget(self.connectUser)
        self.passwordLabel = gui.label(self.connectBox, self, 'Password')
        self.connectPassword = QLineEdit(self.password, self)
        self.connectPassword.setEchoMode(QLineEdit.Password)
        self.connectBox.layout().addWidget(self.connectPassword)
        self.dbLabel = gui.label(self.connectBox, self, 'Database')
        self.connectDB = QLineEdit(self.database, self)
        self.connectBox.layout().addWidget(self.connectDB)
        self.runSQL = gui.auto_commit(self.connectBox, self, 'autocommit',
                                      label='Run SQL', commit=self.commit)
        # query
        self.queryTextEdit = QPlainTextEdit(self.query, self)
        self.sqlBox.layout().addWidget(self.queryTextEdit)
        if self.autocommit:
            self.commit()
    def handleNewSignals(self):
        self._invalidate()
    def commit(self):
        username = self.connectUser.text()
        password = self.connectPassword.text()
        database = self.connectDB.text()
        con = cx_Oracle.connect(username+"/"+password+"@"+database)
        query = self.queryTextEdit.toPlainText()
        cur = con.cursor()
        cur.execute(query)
        results = cur.fetchall()
        columns = [i[0] for i in cur.description]
        df = pandas.DataFrame(results, columns=columns)
        self.send("Dataframe", df)
        orangetable = self.df2table(df)
        #orangedomain = self.df2domain(df)
        self.send("Data", orangetable)
        #self.send("Feature Definitions", orangedomain)
        self.savedQuery = query
        self.savedUsername = username
        self.savedPwd = password
        self.savedDB = database
    def _invalidate(self):
        self.commit()
    def series2descriptor(self, d):
        if d.dtype is np.dtype("float") or d.dtype is np.dtype("int"):
            return ContinuousVariable(str(d.name))
        else:
            t = d.unique()
            #t.sort()
            return DiscreteVariable(str(d.name), list(t.astype("str")))
    def df2domain(self, df):
        featurelist = [self.series2descriptor(df.iloc[:, col]) for col in range(len(df.columns))]
        return Domain(featurelist)
    def df2table(self, df):
        tdomain = self.df2domain(df)
        ttables = [self.series2table(df.iloc[:, i], tdomain[i]) for i in range(len(df.columns))]
        ttables = np.array(ttables).reshape((len(df.columns), -1)).transpose()
        return Table(tdomain, ttables)
    def series2table(self, series, variable):
        if series.dtype is np.dtype("int") or series.dtype is np.dtype("float"):
            series = series.values[:, np.newaxis]
            return Table(series)
        else:
            series = series.astype('category').cat.codes.values.reshape((-1, 1))
            return Table(series)
