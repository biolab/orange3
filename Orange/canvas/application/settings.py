"""
User settings/preference dialog
===============================

"""
import sys
import logging

from AnyQt.QtWidgets import (
    QWidget, QMainWindow, QComboBox, QCheckBox, QListView, QTabWidget,
    QToolBar, QAction, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSizePolicy, QLineEdit, QLabel
)

from AnyQt.QtCore import (
    Qt, QEventLoop, QAbstractItemModel, QModelIndex
)

from .. import config
from ..utils.settings import SettingChangedEvent
from ..utils.propertybindings import (
    AbstractBoundProperty, PropertyBinding, BindingManager
)


log = logging.getLogger(__name__)


def refresh_proxies():
    from Orange.canvas.__main__ import fix_set_proxy_env
    fix_set_proxy_env()


class UserDefaultsPropertyBinding(AbstractBoundProperty):
    """
    A Property binding for a setting in a
    :class:`Orange.canvas.utility.settings.Settings` instance.

    """
    def __init__(self, obj, propertyName, parent=None):
        AbstractBoundProperty.__init__(self, obj, propertyName, parent)

        obj.installEventFilter(self)

    def get(self):
        return self.obj.get(self.propertyName)

    def set(self, value):
        self.obj[self.propertyName] = value

    def eventFilter(self, obj, event):
        if event.type() == SettingChangedEvent.SettingChanged and \
                event.key() == self.propertyName:
            self.notifyChanged()

        return AbstractBoundProperty.eventFilter(self, obj, event)


class UserSettingsModel(QAbstractItemModel):
    """
    An Item Model for user settings presenting a list of
    key, setting value entries along with it's status and type.

    """
    def __init__(self, parent=None, settings=None):
        QAbstractItemModel.__init__(self, parent)

        self.__settings = settings
        self.__headers = ["Name", "Status", "Type", "Value"]

    def setSettings(self, settings):
        if self.__settings != settings:
            self.__settings = settings
            self.reset()

    def settings(self):
        return self.__settings

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        elif self.__settings:
            return len(self.__settings)
        else:
            return 0

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self.__headers)

    def parent(self, index):
        return QModelIndex()

    def index(self, row, column=0, parent=QModelIndex()):
        if parent.isValid() or \
                column < 0 or column >= self.columnCount() or \
                row < 0 or row >= self.rowCount():
            return QModelIndex()

        return self.createIndex(row, column, row)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if section >= 0 and section < 4 and orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.__headers[section]

        return QAbstractItemModel.headerData(self, section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if self._valid(index):
            key = self._keyFromIndex(index)
            column = index.column()
            if role == Qt.DisplayRole:
                if column == 0:
                    return key
                elif column == 1:
                    default = self.__settings.isdefault(key)
                    return "Default" if default else "User"
                elif column == 2:
                    return type(self.__settings.get(key)).__name__
                elif column == 3:
                    return self.__settings.get(key)
                return self

        return None

    def flags(self, index):
        if self._valid(index):
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if index.column() == 3:
                return Qt.ItemIsEditable | flags
            else:
                return flags
        return Qt.NoItemFlags

    def setData(self, index, value, role=Qt.EditRole):
        if self._valid(index) and index.column() == 3:
            key = self._keyFromIndex(index)
            try:
                self.__settings[key] = value
            except (TypeError, ValueError) as ex:
                log.error("Failed to set value (%r) for key %r", value, key,
                          exc_info=True)
            else:
                self.dataChanged.emit(index, index)
                return True

        return False

    def _valid(self, index):
        row = index.row()
        return row >= 0 and row < self.rowCount()

    def _keyFromIndex(self, index):
        row = index.row()
        return list(self.__settings.keys())[row]


def container_widget_helper(orientation=Qt.Vertical, spacing=None, margin=0):
    widget = QWidget()
    if orientation == Qt.Vertical:
        layout = QVBoxLayout()
        widget.setSizePolicy(QSizePolicy.Fixed,
                             QSizePolicy.MinimumExpanding)
    else:
        layout = QHBoxLayout()

    if spacing is not None:
        layout.setSpacing(spacing)

    if margin is not None:
        layout.setContentsMargins(0, 0, 0, 0)

    widget.setLayout(layout)

    return widget


class UserSettingsDialog(QMainWindow):
    """
    A User Settings/Defaults dialog.

    """
    MAC_UNIFIED = True

    def __init__(self, parent=None, **kwargs):
        QMainWindow.__init__(self, parent, **kwargs)
        self.setWindowFlags(Qt.Dialog)
        self.setWindowModality(Qt.ApplicationModal)

        self.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        self.__macUnified = sys.platform == "darwin" and self.MAC_UNIFIED
        self._manager = BindingManager(self,
                                       submitPolicy=BindingManager.AutoSubmit)

        self.__loop = None

        self.__settings = config.settings()
        self.__setupUi()

    def __setupUi(self):
        """Set up the UI.
        """
        if self.__macUnified:
            self.tab = QToolBar()

            self.addToolBar(Qt.TopToolBarArea, self.tab)
            self.setUnifiedTitleAndToolBarOnMac(True)

            # This does not seem to work
            self.setWindowFlags(self.windowFlags() & \
                                ~Qt.MacWindowToolBarButtonHint)

            self.tab.actionTriggered[QAction].connect(
                self.__macOnToolBarAction
            )

            central = QStackedWidget()

            central.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            self.tab = central = QTabWidget(self)

        self.stack = central

        self.setCentralWidget(central)

        # General Tab
        tab = QWidget()
        self.addTab(tab, self.tr("General"),
                    toolTip=self.tr("General Options"))

        form = QFormLayout()
        tab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        nodes = QWidget(self, objectName="nodes")
        nodes.setLayout(QVBoxLayout())
        nodes.layout().setContentsMargins(0, 0, 0, 0)

        cb_anim = QCheckBox(
            self.tr("Enable node animations"),
            objectName="enable-node-animations",
            toolTip=self.tr("Enable shadow and ping animations for nodes "
                            "in the workflow.")
        )
        self.bind(cb_anim, "checked", "schemeedit/enable-node-animations")
        nodes.layout().addWidget(cb_anim)

        form.addRow(self.tr("Nodes"), nodes)

        links = QWidget(self, objectName="links")
        links.setLayout(QVBoxLayout())
        links.layout().setContentsMargins(0, 0, 0, 0)

        cb_show = QCheckBox(
            self.tr("Show channel names between widgets"),
            objectName="show-channel-names",
            toolTip=self.tr("Show source and sink channel names "
                            "over the links.")
        )

        self.bind(cb_show, "checked", "schemeedit/show-channel-names")

        links.layout().addWidget(cb_show)

        form.addRow(self.tr("Links"), links)

        quickmenu = QWidget(self, objectName="quickmenu-options")
        quickmenu.setLayout(QVBoxLayout())
        quickmenu.layout().setContentsMargins(0, 0, 0, 0)

        cb1 = QCheckBox(self.tr("On double click"),
                        toolTip=self.tr("Open quick menu on a double click "
                                        "on an empty spot in the canvas"))

        cb2 = QCheckBox(self.tr("On right click"),
                        toolTip=self.tr("Open quick menu on a right click "
                                        "on an empty spot in the canvas"))

        cb3 = QCheckBox(self.tr("On space key press"),
                        toolTip=self.tr("On Space key press while the mouse"
                                        "is hovering over the canvas."))

        cb4 = QCheckBox(self.tr("On any key press"),
                        toolTip=self.tr("On any key press while the mouse"
                                        "is hovering over the canvas."))

        self.bind(cb1, "checked", "quickmenu/trigger-on-double-click")
        self.bind(cb2, "checked", "quickmenu/trigger-on-right-click")
        self.bind(cb3, "checked", "quickmenu/trigger-on-space-key")
        self.bind(cb4, "checked", "quickmenu/trigger-on-any-key")

        quickmenu.layout().addWidget(cb1)
        quickmenu.layout().addWidget(cb2)
        quickmenu.layout().addWidget(cb3)
        quickmenu.layout().addWidget(cb4)

        form.addRow(self.tr("Open quick menu on"), quickmenu)

        startup = QWidget(self, objectName="startup-group")
        startup.setLayout(QVBoxLayout())
        startup.layout().setContentsMargins(0, 0, 0, 0)

        cb_splash = QCheckBox(self.tr("Show splash screen"), self,
                              objectName="show-splash-screen")

        cb_welcome = QCheckBox(self.tr("Show welcome screen"), self,
                               objectName="show-welcome-screen")

        cb_updates = QCheckBox(self.tr("Check for updates"), self,
                               objectName="check-updates")

        self.bind(cb_splash, "checked", "startup/show-splash-screen")
        self.bind(cb_welcome, "checked", "startup/show-welcome-screen")
        self.bind(cb_updates, "checked", "startup/check-updates")

        startup.layout().addWidget(cb_splash)
        startup.layout().addWidget(cb_welcome)
        startup.layout().addWidget(cb_updates)

        form.addRow(self.tr("On startup"), startup)

        toolbox = QWidget(self, objectName="toolbox-group")
        toolbox.setLayout(QVBoxLayout())
        toolbox.layout().setContentsMargins(0, 0, 0, 0)

        exclusive = QCheckBox(self.tr("Only one tab can be open at a time"))

        self.bind(exclusive, "checked", "mainwindow/toolbox-dock-exclusive")

        toolbox.layout().addWidget(exclusive)

        form.addRow(self.tr("Tool box"), toolbox)
        tab.setLayout(form)

        # Output Tab
        tab = QWidget()
        self.addTab(tab, self.tr("Output"),
                    toolTip="Output Redirection")

        form = QFormLayout()

        box = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.addItems([self.tr("Critical"),
                        self.tr("Error"),
                        self.tr("Warn"),
                        self.tr("Info"),
                        self.tr("Debug")])
        self.bind(combo, "currentIndex", "logging/level")
        layout.addWidget(combo)
        box.setLayout(layout)
        form.addRow(self.tr("Logging"), box)

        box = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        cb1 = QCheckBox(self.tr("Open in external browser"),
                        objectName="open-in-external-browser")
        self.bind(cb1, "checked", "help/open-in-external-browser")
        layout.addWidget(cb1)
        box.setLayout(layout)
        form.addRow(self.tr("Help window"), box)

        tab.setLayout(form)

        # Error Reporting Tab
        tab = QWidget()
        self.addTab(tab, self.tr("Error Reporting"),
                    toolTip="Settings related to error reporting")

        form = QFormLayout()
        line_edit_mid = QLineEdit()
        self.bind(line_edit_mid, "text", "error-reporting/machine-id")
        form.addRow("Machine ID:", line_edit_mid)

        box = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        cb1 = QCheckBox(self.tr(""),
                        toolTip=self.tr("Share anonymous usage statistics to improve Orange"))
        self.bind(cb1, "checked", "error-reporting/send-statistics")
        layout.addWidget(cb1)
        box.setLayout(layout)
        form.addRow(self.tr("Share Anonymous Statistics"), box)

        tab.setLayout(form)

        # Add-ons Tab
        tab = QWidget()
        self.addTab(tab, self.tr("Add-ons"),
                    toolTip="Settings related to add-on installation")

        form = QFormLayout()
        conda = QWidget(self, objectName="conda-group")
        conda.setLayout(QVBoxLayout())
        conda.layout().setContentsMargins(0, 0, 0, 0)

        cb_conda_install = QCheckBox(self.tr("Install add-ons with conda"), self,
                                     objectName="allow-conda")
        self.bind(cb_conda_install, "checked", "add-ons/allow-conda")
        conda.layout().addWidget(cb_conda_install)

        form.addRow(self.tr("Conda"), conda)

        form.addRow(self.tr("Pip"), QLabel("Pip install arguments:"))
        line_edit_pip = QLineEdit()
        self.bind(line_edit_pip, "text", "add-ons/pip-install-arguments")
        form.addRow("", line_edit_pip)

        tab.setLayout(form)

        # Network Tab
        tab = QWidget()
        self.addTab(tab, self.tr("Network"),
                    toolTip="Settings related to networking")

        form = QFormLayout()
        line_edit_http_proxy = QLineEdit()
        self.bind(line_edit_http_proxy, "text", "network/http-proxy")
        form.addRow("HTTP proxy:", line_edit_http_proxy)
        line_edit_https_proxy = QLineEdit()
        self.bind(line_edit_https_proxy, "text", "network/https-proxy")
        form.addRow("HTTPS proxy:", line_edit_https_proxy)
        tab.setLayout(form)

        if self.__macUnified:
            # Need some sensible size otherwise mac unified toolbar 'takes'
            # the space that should be used for layout of the contents
            self.adjustSize()

    def addTab(self, widget, text, toolTip=None, icon=None):
        if self.__macUnified:
            action = QAction(text, self)

            if toolTip:
                action.setToolTip(toolTip)

            if icon:
                action.setIcon(toolTip)
            action.setData(len(self.tab.actions()))

            self.tab.addAction(action)

            self.stack.addWidget(widget)
        else:
            i = self.tab.addTab(widget, text)

            if toolTip:
                self.tab.setTabToolTip(i, toolTip)

            if icon:
                self.tab.setTabIcon(i, icon)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide()
            self.deleteLater()

    def bind(self, source, source_property, key, transformer=None):
        target = UserDefaultsPropertyBinding(self.__settings, key)
        source = PropertyBinding(source, source_property)
        source.set(target.get())

        self._manager.bind(target, source)

    def commit(self):
        self._manager.commit()

    def revert(self):
        self._manager.revert()

    def reset(self):
        for target, source in self._manager.bindings():
            try:
                source.reset()
            except NotImplementedError:
                # Cannot reset.
                pass
            except Exception:
                log.error("Error reseting %r", source.propertyName,
                          exc_info=True)

    def exec_(self):
        self.__loop = QEventLoop()
        self.show()
        status = self.__loop.exec_()
        self.__loop = None
        refresh_proxies()
        return status

    def hideEvent(self, event):
        QMainWindow.hideEvent(self, event)
        if self.__loop is not None:
            self.__loop.exit(0)
            self.__loop = None

    def __macOnToolBarAction(self, action):
        self.stack.setCurrentIndex(action.data())
