"""
Orange Canvas Main Window

"""
import os
import sys
import logging
import operator
from functools import partial
from io import BytesIO

import pkg_resources

from PyQt4.QtGui import (
    QMainWindow, QWidget, QAction, QActionGroup, QMenu, QMenuBar, QDialog,
    QFileDialog, QMessageBox, QVBoxLayout, QSizePolicy, QColor, QKeySequence,
    QIcon, QToolBar, QToolButton, QDockWidget, QDesktopServices, QApplication,
)

from PyQt4.QtCore import (
    Qt, QEvent, QSize, QUrl, QTimer, QFile, QByteArray
)

from PyQt4.QtNetwork import QNetworkDiskCache

from PyQt4.QtWebKit import QWebView

from PyQt4.QtCore import pyqtProperty as Property

# Compatibility with PyQt < v4.8.3
from ..utils.qtcompat import QSettings

from ..gui.dropshadow import DropShadowFrame
from ..gui.dock import CollapsibleDockWidget
from ..gui.quickhelp import QuickHelpTipEvent
from ..gui.utils import message_critical, message_question, \
                        message_warning, message_information

from ..help import HelpManager

from .canvastooldock import CanvasToolDock, QuickCategoryToolbar, \
                            CategoryPopupMenu, popup_position_from_source
from .aboutdialog import AboutDialog
from .schemeinfo import SchemeInfoDialog
from .outputview import OutputView
from .settings import UserSettingsDialog

from ..document.schemeedit import SchemeEditWidget

from ..scheme import widgetsscheme
from ..scheme.readwrite import scheme_load, sniff_version

from . import welcomedialog
from ..preview import previewdialog, previewmodel

from .. import config

from . import tutorials

log = logging.getLogger(__name__)

# TODO: Orange Version in the base link

BASE_LINK = "http://orange.biolab.si/"

LINKS = \
    {"start-using": BASE_LINK + "start-using/",
     "tutorial": BASE_LINK + "tutorial/",
     "reference": BASE_LINK + "doc/"
     }


def style_icons(widget, standard_pixmap):
    """Return the Qt standard pixmap icon.
    """
    return QIcon(widget.style().standardPixmap(standard_pixmap))


def canvas_icons(name):
    """Return the named canvas icon.
    """
    icon_file = QFile("canvas_icons:" + name)
    if icon_file.exists():
        return QIcon("canvas_icons:" + name)
    else:
        return QIcon(pkg_resources.resource_filename(
                      config.__name__,
                      os.path.join("icons", name))
                     )


class FakeToolBar(QToolBar):
    """A Toolbar with no contents (used to reserve top and bottom margins
    on the main window).

    """
    def __init__(self, *args, **kwargs):
        QToolBar.__init__(self, *args, **kwargs)
        self.setFloatable(False)
        self.setMovable(False)

        # Don't show the tool bar action in the main window's
        # context menu.
        self.toggleViewAction().setVisible(False)

    def paintEvent(self, event):
        # Do nothing.
        pass


class DockableWindow(QDockWidget):
    def __init__(self, *args, **kwargs):
        QDockWidget.__init__(self, *args, **kwargs)

        # Fist show after floating
        self.__firstShow = True
        # Flags to use while floating
        self.__windowFlags = Qt.Window
        self.setWindowFlags(self.__windowFlags)
        self.topLevelChanged.connect(self.__on_topLevelChanged)
        self.visibilityChanged.connect(self.__on_visbilityChanged)

        self.__closeAction = QAction(self.tr("Close"), self,
                                     shortcut=QKeySequence.Close,
                                     triggered=self.close,
                                     enabled=self.isFloating())
        self.topLevelChanged.connect(self.__closeAction.setEnabled)
        self.addAction(self.__closeAction)

    def setFloatingWindowFlags(self, flags):
        """
        Set `windowFlags` to use while the widget is floating (undocked).
        """
        if self.__windowFlags != flags:
            self.__windowFlags = flags
            if self.isFloating():
                self.__fixWindowFlags()

    def floatingWindowFlags(self):
        """
        Return the `windowFlags` used when the widget is floating.
        """
        return self.__windowFlags

    def __fixWindowFlags(self):
        if self.isFloating():
            update_window_flags(self, self.__windowFlags)

    def __on_topLevelChanged(self, floating):
        if floating:
            self.__firstShow = True
            self.__fixWindowFlags()

    def __on_visbilityChanged(self, visible):
        if visible and self.isFloating() and self.__firstShow:
            self.__firstShow = False
            self.__fixWindowFlags()


def update_window_flags(widget, flags):
    currflags = widget.windowFlags()
    if int(flags) != int(currflags):
        hidden = widget.isHidden()
        widget.setWindowFlags(flags)
        # setting the flags hides the widget
        if not hidden:
            widget.show()


class CanvasMainWindow(QMainWindow):
    SETTINGS_VERSION = 2

    def __init__(self, *args):
        QMainWindow.__init__(self, *args)

        self.__scheme_margins_enabled = True
        self.__document_title = "untitled"
        self.__first_show = True

        self.widget_registry = None

        self.last_scheme_dir = QDesktopServices.StandardLocation(
            QDesktopServices.DocumentsLocation
        )
        try:
            self.recent_schemes = config.recent_schemes()
        except Exception:
            log.error("Failed to load recent scheme list.", exc_info=True)
            self.recent_schemes = []

        self.num_recent_schemes = 15

        self.open_in_external_browser = False
        self.help = HelpManager(self)

        self.setup_actions()
        self.setup_ui()
        self.setup_menu()

        self.restore()

    def setup_ui(self):
        """Setup main canvas ui
        """

        log.info("Setting up Canvas main window.")

        # Two dummy tool bars to reserve space
        self.__dummy_top_toolbar = FakeToolBar(
                            objectName="__dummy_top_toolbar")
        self.__dummy_bottom_toolbar = FakeToolBar(
                            objectName="__dummy_bottom_toolbar")

        self.__dummy_top_toolbar.setFixedHeight(20)
        self.__dummy_bottom_toolbar.setFixedHeight(20)

        self.addToolBar(Qt.TopToolBarArea, self.__dummy_top_toolbar)
        self.addToolBar(Qt.BottomToolBarArea, self.__dummy_bottom_toolbar)

        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        self.setDockOptions(QMainWindow.AnimatedDocks)
        # Create an empty initial scheme inside a container with fixed
        # margins.
        w = QWidget()
        w.setLayout(QVBoxLayout())
        w.layout().setContentsMargins(20, 0, 10, 0)

        self.scheme_widget = SchemeEditWidget()
        self.scheme_widget.setScheme(widgetsscheme.WidgetsScheme(parent=self))

        w.layout().addWidget(self.scheme_widget)

        self.setCentralWidget(w)

        # Drop shadow around the scheme document
        frame = DropShadowFrame(radius=15)
        frame.setColor(QColor(0, 0, 0, 100))
        frame.setWidget(self.scheme_widget)

        # Main window title and title icon.
        self.set_document_title(self.scheme_widget.scheme().title)
        self.scheme_widget.titleChanged.connect(self.set_document_title)
        self.scheme_widget.modificationChanged.connect(self.setWindowModified)

        # QMainWindow's Dock widget
        self.dock_widget = CollapsibleDockWidget(objectName="main-area-dock")
        self.dock_widget.setFeatures(QDockWidget.DockWidgetMovable | \
                                     QDockWidget.DockWidgetClosable)

        self.dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | \
                                         Qt.RightDockWidgetArea)

        # Main canvas tool dock (with widget toolbox, common actions.
        # This is the widget that is shown when the dock is expanded.
        canvas_tool_dock = CanvasToolDock(objectName="canvas-tool-dock")
        canvas_tool_dock.setSizePolicy(QSizePolicy.Fixed,
                                       QSizePolicy.MinimumExpanding)

        # Bottom tool bar
        self.canvas_toolbar = canvas_tool_dock.toolbar
        self.canvas_toolbar.setIconSize(QSize(25, 25))
        self.canvas_toolbar.setFixedHeight(28)
        self.canvas_toolbar.layout().setSpacing(1)

        # Widgets tool box
        self.widgets_tool_box = canvas_tool_dock.toolbox
        self.widgets_tool_box.setObjectName("canvas-toolbox")
        self.widgets_tool_box.setTabButtonHeight(30)
        self.widgets_tool_box.setTabIconSize(QSize(26, 26))
        self.widgets_tool_box.setButtonSize(QSize(64, 84))
        self.widgets_tool_box.setIconSize(QSize(48, 48))

        self.widgets_tool_box.triggered.connect(
            self.on_tool_box_widget_activated
        )

        self.dock_help = canvas_tool_dock.help
        self.dock_help.setMaximumHeight(150)
        self.dock_help.document().setDefaultStyleSheet("h3, a {color: orange;}")

        self.dock_help_action = canvas_tool_dock.toogleQuickHelpAction()
        self.dock_help_action.setText(self.tr("Show Help"))
        self.dock_help_action.setIcon(canvas_icons("Info.svg"))

        self.canvas_tool_dock = canvas_tool_dock

        # Dock contents when collapsed (a quick category tool bar, ...)
        dock2 = QWidget(objectName="canvas-quick-dock")
        dock2.setLayout(QVBoxLayout())
        dock2.layout().setContentsMargins(0, 0, 0, 0)
        dock2.layout().setSpacing(0)
        dock2.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        self.quick_category = QuickCategoryToolbar()
        self.quick_category.setButtonSize(QSize(38, 30))
        self.quick_category.actionTriggered.connect(
            self.on_quick_category_action
        )

        tool_actions = self.current_document().toolbarActions()

        (self.canvas_zoom_action, self.canvas_align_to_grid_action,
         self.canvas_text_action, self.canvas_arrow_action,) = tool_actions

        self.canvas_zoom_action.setIcon(canvas_icons("Search.svg"))
        self.canvas_align_to_grid_action.setIcon(canvas_icons("Grid.svg"))
        self.canvas_text_action.setIcon(canvas_icons("Text Size.svg"))
        self.canvas_arrow_action.setIcon(canvas_icons("Arrow.svg"))

        dock_actions = [self.show_properties_action] + \
                       tool_actions + \
                       [self.freeze_action,
                        self.dock_help_action]

        # Tool bar in the collapsed dock state (has the same actions as
        # the tool bar in the CanvasToolDock
        actions_toolbar = QToolBar(orientation=Qt.Vertical)
        actions_toolbar.setFixedWidth(38)
        actions_toolbar.layout().setSpacing(0)

        actions_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)

        for action in dock_actions:
            self.canvas_toolbar.addAction(action)
            button = self.canvas_toolbar.widgetForAction(action)
            button.setPopupMode(QToolButton.DelayedPopup)

            actions_toolbar.addAction(action)
            button = actions_toolbar.widgetForAction(action)
            button.setFixedSize(38, 30)
            button.setPopupMode(QToolButton.DelayedPopup)

        dock2.layout().addWidget(self.quick_category)
        dock2.layout().addWidget(actions_toolbar)

        self.dock_widget.setAnimationEnabled(False)
        self.dock_widget.setExpandedWidget(self.canvas_tool_dock)
        self.dock_widget.setCollapsedWidget(dock2)
        self.dock_widget.setExpanded(True)
        self.dock_widget.expandedChanged.connect(self._on_tool_dock_expanded)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget)
        self.dock_widget.dockLocationChanged.connect(
            self._on_dock_location_changed
        )

        self.output_dock = DockableWindow(self.tr("Output"), self,
                                          objectName="output-dock")
        self.output_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        output_view = OutputView()
        # Set widget before calling addDockWidget, otherwise the dock
        # does not resize properly on first undock
        self.output_dock.setWidget(output_view)
        self.output_dock.hide()

        self.help_dock = DockableWindow(self.tr("Help"), self,
                                        objectName="help-dock")
        self.help_dock.setAllowedAreas(Qt.NoDockWidgetArea)
        self.help_view = QWebView()
        manager = self.help_view.page().networkAccessManager()
        cache = QNetworkDiskCache()
        cache.setCacheDirectory(
            os.path.join(config.cache_dir(), "help", "help-view-cache")
        )
        manager.setCache(cache)
        self.help_dock.setWidget(self.help_view)
        self.help_dock.hide()

        self.setMinimumSize(600, 500)

    def setup_actions(self):
        """Initialize main window actions.
        """

        self.new_action = \
            QAction(self.tr("New"), self,
                    objectName="action-new",
                    toolTip=self.tr("Open a new workflow."),
                    triggered=self.new_scheme,
                    shortcut=QKeySequence.New,
                    icon=canvas_icons("New.svg")
                    )

        self.open_action = \
            QAction(self.tr("Open"), self,
                    objectName="action-open",
                    toolTip=self.tr("Open a workflow."),
                    triggered=self.open_scheme,
                    shortcut=QKeySequence.Open,
                    icon=canvas_icons("Open.svg")
                    )

        self.open_and_freeze_action = \
            QAction(self.tr("Open and Freeze"), self,
                    objectName="action-open-and-freeze",
                    toolTip=self.tr("Open a new workflow and freeze signal "
                                    "propagation."),
                    triggered=self.open_and_freeze_scheme
                    )

        self.open_report_action = \
            QAction(self.tr("Open Report"), self,
                    objectName="action-open-report",
                    triggered=self.open_report,
                    )

        self.save_action = \
            QAction(self.tr("Save"), self,
                    objectName="action-save",
                    toolTip=self.tr("Save current workflow."),
                    triggered=self.save_scheme,
                    shortcut=QKeySequence.Save,
                    )

        self.save_as_action = \
            QAction(self.tr("Save As ..."), self,
                    objectName="action-save-as",
                    toolTip=self.tr("Save current workflow as."),
                    triggered=self.save_scheme_as,
                    shortcut=QKeySequence.SaveAs,
                    )

        self.quit_action = \
            QAction(self.tr("Quit"), self,
                    objectName="quit-action",
                    toolTip=self.tr("Quit Orange Canvas."),
                    triggered=self.quit,
                    menuRole=QAction.QuitRole,
                    shortcut=QKeySequence.Quit,
                    )

        self.welcome_action = \
            QAction(self.tr("Welcome"), self,
                    objectName="welcome-action",
                    toolTip=self.tr("Show welcome screen."),
                    triggered=self.welcome_dialog,
                    )

        self.get_started_action = \
            QAction(self.tr("Get Started"), self,
                    objectName="get-started-action",
                    toolTip=self.tr("View a 'Get Started' introduction."),
                    triggered=self.get_started,
                    icon=canvas_icons("Get Started.svg")
                    )

        self.tutorials_action = \
            QAction(self.tr("Tutorials"), self,
                    objectName="tutorial-action",
                    toolTip=self.tr("Browse tutorials."),
                    triggered=self.tutorial_scheme,
                    icon=canvas_icons("Tutorials.svg")
                    )

        self.documentation_action = \
            QAction(self.tr("Documentation"), self,
                    objectName="documentation-action",
                    toolTip=self.tr("View reference documentation."),
                    triggered=self.documentation,
                    icon=canvas_icons("Documentation.svg")
                    )

        self.about_action = \
            QAction(self.tr("About"), self,
                    objectName="about-action",
                    toolTip=self.tr("Show about dialog."),
                    triggered=self.open_about,
                    menuRole=QAction.AboutRole,
                    )

        # Action group for for recent scheme actions
        self.recent_scheme_action_group = \
            QActionGroup(self, exclusive=False,
                         objectName="recent-action-group",
                         triggered=self._on_recent_scheme_action)

        self.recent_action = \
            QAction(self.tr("Browse Recent"), self,
                    objectName="recent-action",
                    toolTip=self.tr("Browse and open a recent workflow."),
                    triggered=self.recent_scheme,
                    shortcut=QKeySequence(Qt.ControlModifier | \
                                          (Qt.ShiftModifier | Qt.Key_R)),
                    icon=canvas_icons("Recent.svg")
                    )

        self.reload_last_action = \
            QAction(self.tr("Reload Last Workflow"), self,
                    objectName="reload-last-action",
                    toolTip=self.tr("Reload last open workflow."),
                    triggered=self.reload_last,
                    shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_R)
                    )

        self.clear_recent_action = \
            QAction(self.tr("Clear Menu"), self,
                    objectName="clear-recent-menu-action",
                    toolTip=self.tr("Clear recent menu."),
                    triggered=self.clear_recent_schemes
                    )

        self.show_properties_action = \
            QAction(self.tr("Workflow Info"), self,
                    objectName="show-properties-action",
                    toolTip=self.tr("Show workflow properties."),
                    triggered=self.show_scheme_properties,
                    shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_I),
                    icon=canvas_icons("Document Info.svg")
                    )

        self.canvas_settings_action = \
            QAction(self.tr("Settings"), self,
                    objectName="canvas-settings-action",
                    toolTip=self.tr("Set application settings."),
                    triggered=self.open_canvas_settings,
                    menuRole=QAction.PreferencesRole,
                    shortcut=QKeySequence.Preferences
                    )

        self.canvas_addons_action = \
            QAction(self.tr("&Add-ons..."), self,
                    objectName="canvas-addons-action",
                    toolTip=self.tr("Manage add-ons."),
                    triggered=self.open_addons,
                    )

        self.show_output_action = \
            QAction(self.tr("Show Output View"), self,
                    toolTip=self.tr("Show application output."),
                    triggered=self.show_output_view,
                    )

        self.show_report_action = \
            QAction(self.tr("Show Report View"), self,
                    triggered=self.show_report_view,
                    shortcut=QKeySequence(Qt.ShiftModifier | Qt.Key_R)
                    )

        if sys.platform == "darwin":
            # Actions for native Mac OSX look and feel.
            self.minimize_action = \
                QAction(self.tr("Minimize"), self,
                        triggered=self.showMinimized,
                        shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_M)
                        )

            self.zoom_action = \
                QAction(self.tr("Zoom"), self,
                        objectName="application-zoom",
                        triggered=self.toggleMaximized,
                        )

        self.freeze_action = \
            QAction(self.tr("Freeze"), self,
                    objectName="signal-freeze-action",
                    checkable=True,
                    toolTip=self.tr("Freeze signal propagation."),
                    triggered=self.set_signal_freeze,
                    icon=canvas_icons("Pause.svg")
                    )

        self.toggle_tool_dock_expand = \
            QAction(self.tr("Expand Tool Dock"), self,
                    objectName="toggle-tool-dock-expand",
                    checkable=True,
                    shortcut=QKeySequence(Qt.ControlModifier |
                                          (Qt.ShiftModifier | Qt.Key_D)),
                    triggered=self.set_tool_dock_expanded)
        self.toggle_tool_dock_expand.setChecked(True)

        # Gets assigned in setup_ui (the action is defined in CanvasToolDock)
        # TODO: This is bad (should be moved here).
        self.dock_help_action = None

        self.toogle_margins_action = \
            QAction(self.tr("Show Workflow Margins"), self,
                    checkable=True,
                    toolTip=self.tr("Show margins around the workflow view."),
                    )
        self.toogle_margins_action.setChecked(True)
        self.toogle_margins_action.toggled.connect(
            self.set_scheme_margins_enabled)

        self.reset_widget_settings_action = \
            QAction(self.tr("Reset widget settings..."), self,
                    triggered=self.reset_widget_settings)

    def setup_menu(self):
        menu_bar = QMenuBar()

        # File menu
        file_menu = QMenu(self.tr("&File"), menu_bar)
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_and_freeze_action)
        file_menu.addAction(self.reload_last_action)

        # File -> Open Recent submenu
        self.recent_menu = QMenu(self.tr("Open Recent"), file_menu)
        file_menu.addMenu(self.recent_menu)
        file_menu.addAction(self.open_report_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.show_properties_action)
        file_menu.addAction(self.quit_action)

        self.recent_menu.addAction(self.recent_action)

        # Store the reference to separator for inserting recent
        # schemes into the menu in `add_recent_scheme`.
        self.recent_menu_begin = self.recent_menu.addSeparator()

        # Add recent items.
        for title, filename in self.recent_schemes:
            action = QAction(title or self.tr("untitled"), self,
                             toolTip=filename)

            action.setData(filename)
            self.recent_menu.addAction(action)
            self.recent_scheme_action_group.addAction(action)

        self.recent_menu.addSeparator()
        self.recent_menu.addAction(self.clear_recent_action)
        menu_bar.addMenu(file_menu)

        editor_menus = self.scheme_widget.menuBarActions()

        # WARNING: Hard coded order, should lookup the action text
        # and determine the proper order
        self.edit_menu = editor_menus[0].menu()
        self.widget_menu = editor_menus[1].menu()

        # Edit menu
        menu_bar.addMenu(self.edit_menu)

        # View menu
        self.view_menu = QMenu(self.tr("&View"), self)
        self.toolbox_menu = QMenu(self.tr("Widget Toolbox Style"),
                                  self.view_menu)
        self.toolbox_menu_group = \
            QActionGroup(self, objectName="toolbox-menu-group")

        self.view_menu.addAction(self.toggle_tool_dock_expand)

        self.view_menu.addSeparator()
        self.view_menu.addAction(self.toogle_margins_action)
        menu_bar.addMenu(self.view_menu)

        # Options menu
        self.options_menu = QMenu(self.tr("&Options"), self)
        self.options_menu.addAction(self.show_output_action)
        self.options_menu.addAction(self.show_report_action)
#        self.options_menu.addAction("Add-ons")
#        self.options_menu.addAction("Developers")
#        self.options_menu.addAction("Run Discovery")
#        self.options_menu.addAction("Show Canvas Log")
#        self.options_menu.addAction("Attach Python Console")
        self.options_menu.addSeparator()
        self.options_menu.addAction(self.canvas_settings_action)
        self.options_menu.addAction(self.reset_widget_settings_action)
        self.options_menu.addAction(self.canvas_addons_action)

        # Widget menu
        menu_bar.addMenu(self.widget_menu)

        if sys.platform == "darwin":
            # Mac OS X native look and feel.
            self.window_menu = QMenu(self.tr("Window"), self)
            self.window_menu.addAction(self.minimize_action)
            self.window_menu.addAction(self.zoom_action)
            menu_bar.addMenu(self.window_menu)

        menu_bar.addMenu(self.options_menu)

        # Help menu.
        self.help_menu = QMenu(self.tr("&Help"), self)
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.welcome_action)
        self.help_menu.addAction(self.tutorials_action)
        self.help_menu.addAction(self.documentation_action)
        menu_bar.addMenu(self.help_menu)

        self.setMenuBar(menu_bar)

    def restore(self):
        """Restore the main window state from saved settings.
        """
        QSettings.setDefaultFormat(QSettings.IniFormat)
        settings = QSettings()
        settings.beginGroup("mainwindow")

        self.dock_widget.setExpanded(
            settings.value("canvasdock/expanded", True, type=bool)
        )

        floatable = settings.value("toolbox-dock-floatable", False, type=bool)
        if floatable:
            self.dock_widget.setFeatures(self.dock_widget.features() | \
                                         QDockWidget.DockWidgetFloatable)

        self.widgets_tool_box.setExclusive(
            settings.value("toolbox-dock-exclusive", True, type=bool)
        )

        self.toogle_margins_action.setChecked(
            settings.value("scheme-margins-enabled", False, type=bool)
        )

        default_dir = QDesktopServices.storageLocation(
            QDesktopServices.DocumentsLocation
        )

        self.last_scheme_dir = settings.value("last-scheme-dir", default_dir,
                                              type=str)

        if not os.path.exists(self.last_scheme_dir):
            # if directory no longer exists reset the saved location.
            self.last_scheme_dir = default_dir

        self.canvas_tool_dock.setQuickHelpVisible(
            settings.value("quick-help/visible", True, type=bool)
        )

        self.__update_from_settings()

    def set_document_title(self, title):
        """Set the document title (and the main window title). If `title`
        is an empty string a default 'untitled' placeholder will be used.

        """
        if self.__document_title != title:
            self.__document_title = title

            if not title:
                # TODO: should the default name be platform specific
                title = self.tr("untitled")

            self.setWindowTitle(title + "[*]")

    def document_title(self):
        """Return the document title.
        """
        return self.__document_title

    def set_widget_registry(self, widget_registry):
        """Set widget registry.
        """
        if self.widget_registry is not None:
            # Clear the dock widget and popup.
            pass

        self.widget_registry = widget_registry
        self.widgets_tool_box.setModel(widget_registry.model())
        self.quick_category.setModel(widget_registry.model())

        self.scheme_widget.setRegistry(widget_registry)

        self.help.set_registry(widget_registry)

        # Restore possibly saved widget toolbox tab states
        settings = QSettings()

        state = settings.value("mainwindow/widgettoolbox/state",
                                defaultValue=QByteArray(),
                                type=QByteArray)
        if state:
            self.widgets_tool_box.restoreState(state)

    def set_quick_help_text(self, text):
        self.canvas_tool_dock.help.setText(text)

    def current_document(self):
        return self.scheme_widget

    def on_tool_box_widget_activated(self, action):
        """A widget action in the widget toolbox has been activated.
        """
        widget_desc = action.data()
        if widget_desc:
            scheme_widget = self.current_document()
            if scheme_widget:
                scheme_widget.createNewNode(widget_desc)

    def on_quick_category_action(self, action):
        """The quick category menu action triggered.
        """
        category = action.text()
        if self.use_popover:
            # Show a popup menu with the widgets in the category
            popup = CategoryPopupMenu(self.quick_category)
            reg = self.widget_registry.model()
            i = index(self.widget_registry.categories(), category,
                      predicate=lambda name, cat: cat.name == name)
            if i != -1:
                popup.setCategoryItem(reg.item(i))
                button = self.quick_category.buttonForAction(action)
                pos = popup_position_from_source(popup, button)
                action = popup.exec_(pos)
                if action is not None:
                    self.on_tool_box_widget_activated(action)

        else:
            for i in range(self.widgets_tool_box.count()):
                cat_act = self.widgets_tool_box.tabAction(i)
                cat_act.setChecked(cat_act.text() == category)

            self.dock_widget.expand()

    def set_scheme_margins_enabled(self, enabled):
        """Enable/disable the margins around the scheme document.
        """
        if self.__scheme_margins_enabled != enabled:
            self.__scheme_margins_enabled = enabled
            self.__update_scheme_margins()

    def scheme_margins_enabled(self):
        return self.__scheme_margins_enabled

    scheme_margins_enabled = Property(bool,
                                      fget=scheme_margins_enabled,
                                      fset=set_scheme_margins_enabled)

    def __update_scheme_margins(self):
        """Update the margins around the scheme document.
        """
        enabled = self.__scheme_margins_enabled
        self.__dummy_top_toolbar.setVisible(enabled)
        self.__dummy_bottom_toolbar.setVisible(enabled)
        central = self.centralWidget()

        margin = 20 if enabled else 0

        if self.dockWidgetArea(self.dock_widget) == Qt.LeftDockWidgetArea:
            margins = (margin / 2, 0, margin, 0)
        else:
            margins = (margin, 0, margin / 2, 0)

        central.layout().setContentsMargins(*margins)

    #################
    # Action handlers
    #################
    def new_scheme(self):
        """New scheme. Return QDialog.Rejected if the user canceled
        the operation and QDialog.Accepted otherwise.

        """
        document = self.current_document()
        if document.isModifiedStrict():
            # Ask for save changes
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        new_scheme = widgetsscheme.WidgetsScheme(parent=self)

        settings = QSettings()
        show = settings.value("schemeinfo/show-at-new-scheme", True,
                              type=bool)

        if show:
            status = self.show_scheme_properties_for(
                new_scheme, self.tr("New Workflow")
            )

            if status == QDialog.Rejected:
                return QDialog.Rejected

        self.set_new_scheme(new_scheme)

        return QDialog.Accepted

    def open_scheme(self):
        """Open a new scheme. Return QDialog.Rejected if the user canceled
        the operation and QDialog.Accepted otherwise.

        """
        document = self.current_document()
        if document.isModifiedStrict():
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        if self.last_scheme_dir is None:
            # Get user 'Documents' folder
            start_dir = QDesktopServices.storageLocation(
                            QDesktopServices.DocumentsLocation)
        else:
            start_dir = self.last_scheme_dir

        # TODO: Use a dialog instance and use 'addSidebarUrls' to
        # set one or more extra sidebar locations where Schemes are stored.
        # Also use setHistory
        filename = QFileDialog.getOpenFileName(
            self, self.tr("Open Orange Workflow File"),
            start_dir, self.tr("Orange Workflow (*.ows)"),
        )

        if filename:
            self.load_scheme(filename)
            return QDialog.Accepted
        else:
            return QDialog.Rejected

    def open_report(self):
        from Orange.canvas.report.owreport import OWReport
        rep = OWReport()
        rep.open_report()

    def open_and_freeze_scheme(self):
        """
        Open a new scheme and freeze signal propagation. Return
        QDialog.Rejected if the user canceled the operation and
        QDialog.Accepted otherwise.

        """
        frozen = self.freeze_action.isChecked()
        if not frozen:
            self.freeze_action.trigger()

        state = self.open_scheme()
        if state == QDialog.Rejected:
            # If the action was rejected restore the original frozen state
            if not frozen:
                self.freeze_action.trigger()
        return state

    def open_scheme_file(self, filename):
        """
        Open and load a scheme file.
        """
        if isinstance(filename, QUrl):
            filename = filename.toLocalFile()

        document = self.current_document()
        if document.isModifiedStrict():
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        self.load_scheme(filename)
        return QDialog.Accepted

    def load_scheme(self, filename):
        """Load a scheme from a file (`filename`) into the current
        document updates the recent scheme list and the loaded scheme path
        property.

        """
        dirname = os.path.dirname(filename)

        self.last_scheme_dir = dirname

        new_scheme = self.new_scheme_from(filename)
        if new_scheme is not None:
            self.set_new_scheme(new_scheme)

            scheme_doc_widget = self.current_document()
            scheme_doc_widget.setPath(filename)

            self.add_recent_scheme(new_scheme.title, filename)

    def load_scheme_xml(self, xml):
        new_scheme = widgetsscheme.WidgetsScheme(parent=self)
        scheme_load(new_scheme, xml)
        self.set_new_scheme(new_scheme)
        return QDialog.Accepted

    def get_scheme_xml(self):
        buffer = BytesIO()
        curr_scheme = self.current_document().scheme()
        curr_scheme.save_to(buffer, pretty=True, pickle_fallback=True)
        return buffer.getvalue().decode("utf-8")

    def new_scheme_from(self, filename):
        """Create and return a new :class:`widgetsscheme.WidgetsScheme`
        from a saved `filename`. Return `None` if an error occurs.

        """
        new_scheme = widgetsscheme.WidgetsScheme(
            parent=self, env={"basedir": os.path.dirname(filename)})
        errors = []
        try:
            scheme_load(new_scheme, open(filename, "rb"),
                        error_handler=errors.append)

        except Exception:
            message_critical(
                 self.tr("Could not load an Orange Workflow file"),
                 title=self.tr("Error"),
                 informative_text=self.tr("An unexpected error occurred "
                                          "while loading '%s'.") % filename,
                 exc_info=True,
                 parent=self)
            return None
        if errors:
            message_warning(
                self.tr("Errors occurred while loading the workflow."),
                title=self.tr("Problem"),
                informative_text=self.tr(
                     "There were problems loading some "
                     "of the widgets/links in the "
                     "workflow."
                ),
                details="\n".join(map(repr, errors))
            )
        return new_scheme

    def reload_last(self):
        """Reload last opened scheme. Return QDialog.Rejected if the
        user canceled the operation and QDialog.Accepted otherwise.

        """
        document = self.current_document()
        if document.isModifiedStrict():
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        # TODO: Search for a temp backup scheme with per process
        # locking.
        if self.recent_schemes:
            self.load_scheme(self.recent_schemes[0][1])

        return QDialog.Accepted

    def set_new_scheme(self, new_scheme):
        """
        Set new_scheme as the current shown scheme. The old scheme
        will be deleted.

        """
        scheme_doc = self.current_document()
        old_scheme = scheme_doc.scheme()

        manager = new_scheme.signal_manager
        if self.freeze_action.isChecked():
            manager.pause()

        scheme_doc.setScheme(new_scheme)

        # Send a close event to the Scheme, it is responsible for
        # closing/clearing all resources (widgets).
        QApplication.sendEvent(old_scheme, QEvent(QEvent.Close))

        old_scheme.deleteLater()

    def ask_save_changes(self):
        """Ask the user to save the changes to the current scheme.
        Return QDialog.Accepted if the scheme was successfully saved
        or the user selected to discard the changes. Otherwise return
        QDialog.Rejected.

        """
        document = self.current_document()
        title = document.scheme().title or "untitled"
        selected = message_question(
            self.tr('Do you want to save the changes you made to workflow "%s"?')
                    % title,
            self.tr("Save Changes?"),
            self.tr("Your changes will be lost if you do not save them."),
            buttons=QMessageBox.Save | QMessageBox.Cancel | \
                    QMessageBox.Discard,
            default_button=QMessageBox.Save,
            parent=self)

        if selected == QMessageBox.Save:
            return self.save_scheme()
        elif selected == QMessageBox.Discard:
            return QDialog.Accepted
        elif selected == QMessageBox.Cancel:
            return QDialog.Rejected

    def check_can_save(self, document, path):
        """
        Check if saving the document to `path` would prevent it from
        being read by the version 1.0 of scheme parser. Return ``True``
        if the existing scheme is version 1.0 else show a message box and
        return ``False``

        .. note::
            In case of an error (opening, parsing), this method will return
            ``True``, so the

        """
        if path and os.path.exists(path):
            try:
                version = sniff_version(open(path, "rb"))
            except (IOError, OSError):
                log.error("Error opening '%s'", path, exc_info=True)
                # The client should fail attempting to write.
                return True
            except Exception:
                log.error("Error sniffing workflow version in '%s'", path,
                          exc_info=True)
                # Malformed .ows file, ...
                return True

            if version == "1.0":
                # TODO: Ask for overwrite confirmation instead
                message_information(
                    self.tr("Can not overwrite a version 1.0 ows file. "
                            "Please save your work to a new file"),
                    title="Info",
                    parent=self)
                return False
        return True

    def save_scheme(self):
        """Save the current scheme. If the scheme does not have an associated
        path then prompt the user to select a scheme file. Return
        QDialog.Accepted if the scheme was successfully saved and
        QDialog.Rejected if the user canceled the file selection.

        """
        document = self.current_document()
        curr_scheme = document.scheme()
        path = document.path()

        if path and self.check_can_save(document, path):
            if self.save_scheme_to(curr_scheme, path):
                document.setModified(False)
                self.add_recent_scheme(curr_scheme.title, document.path())
                return QDialog.Accepted
            else:
                return QDialog.Rejected
        else:
            return self.save_scheme_as()

    def save_scheme_as(self):
        """
        Save the current scheme by asking the user for a filename. Return
        `QFileDialog.Accepted` if the scheme was saved successfully and
        `QFileDialog.Rejected` if not.

        """
        document = self.current_document()
        curr_scheme = document.scheme()
        title = curr_scheme.title or "untitled"

        if document.path():
            start_dir = document.path()
        else:
            if self.last_scheme_dir is not None:
                start_dir = self.last_scheme_dir
            else:
                start_dir = QDesktopServices.storageLocation(
                    QDesktopServices.DocumentsLocation
                )

            start_dir = os.path.join(str(start_dir), title + ".ows")

        filename = QFileDialog.getSaveFileName(
            self, self.tr("Save Orange Workflow File"),
            start_dir, self.tr("Orange Workflow (*.ows)")
        )

        if filename:
            if not self.check_can_save(document, filename):
                return QDialog.Rejected

            self.last_scheme_dir = os.path.dirname(filename)

            if self.save_scheme_to(curr_scheme, filename):
                document.setPath(filename)
                document.setModified(False)
                self.add_recent_scheme(curr_scheme.title, document.path())

                return QFileDialog.Accepted

        return QFileDialog.Rejected

    def save_scheme_to(self, scheme, filename):
        """
        Save a Scheme instance `scheme` to `filename`. On success return
        `True`, else show a message to the user explaining the error and
        return `False`.

        """
        dirname, basename = os.path.split(filename)
        self.last_scheme_dir = dirname
        title = scheme.title or "untitled"

        # First write the scheme to a buffer so we don't truncate an
        # existing scheme file if `scheme.save_to` raises an error.
        buffer = BytesIO()
        try:
            scheme.save_to(buffer, pretty=True, pickle_fallback=True)
        except Exception:
            log.error("Error saving %r to %r", scheme, filename, exc_info=True)
            message_critical(
                self.tr('An error occurred while trying to save workflow '
                        '"%s" to "%s"') % (title, basename),
                title=self.tr("Error saving %s") % basename,
                exc_info=True,
                parent=self
            )
            return False

        try:
            with open(filename, "wb") as f:
                f.write(buffer.getvalue())
            scheme.set_runtime_env("basedir", os.path.dirname(filename))
            return True
        except (IOError, OSError) as ex:
            log.error("%s saving '%s'", type(ex).__name__, filename,
                      exc_info=True)
            if ex.errno == 2:
                # user might enter a string containing a path separator
                message_warning(
                    self.tr('Workflow "%s" could not be saved. The path does '
                            'not exist') % title,
                    title="",
                    informative_text=self.tr("Choose another location."),
                    parent=self
                )
            elif ex.errno == 13:
                message_warning(
                    self.tr('Workflow "%s" could not be saved. You do not '
                            'have write permissions.') % title,
                    title="",
                    informative_text=self.tr(
                        "Change the file system permissions or choose "
                        "another location."),
                    parent=self
                )
            else:
                message_warning(
                    self.tr('Workflow "%s" could not be saved.') % title,
                    title="",
                    informative_text=ex.strerror,
                    exc_info=True,
                    parent=self
                )
            return False

        except Exception:
            log.error("Error saving %r to %r", scheme, filename, exc_info=True)
            message_critical(
                self.tr('An error occurred while trying to save workflow '
                        '"%s" to "%s"') % (title, basename),
                title=self.tr("Error saving %s") % basename,
                exc_info=True,
                parent=self
            )
            return False

    def get_started(self, *args):
        """Show getting started video
        """
        url = QUrl(LINKS["start-using"])
        QDesktopServices.openUrl(url)

    def tutorial(self, *args):
        """Show tutorial.
        """
        url = QUrl(LINKS["tutorial"])
        QDesktopServices.openUrl(url)

    def documentation(self, *args):
        """Show reference documentation.
        """
        url = QUrl(LINKS["tutorial"])
        QDesktopServices.openUrl(url)

    def recent_scheme(self, *args):
        """Browse recent schemes. Return QDialog.Rejected if the user
        canceled the operation and QDialog.Accepted otherwise.

        """
        items = [previewmodel.PreviewItem(name=title, path=path)
                 for title, path in self.recent_schemes]
        model = previewmodel.PreviewModel(items=items)

        dialog = previewdialog.PreviewDialog(self)
        title = self.tr("Recent Workflows")
        dialog.setWindowTitle(title)
        template = ('<h3 style="font-size: 26px">\n'
                    #'<img height="26" src="canvas_icons:Recent.svg">\n'
                    '{0}\n'
                    '</h3>')
        dialog.setHeading(template.format(title))
        dialog.setModel(model)

        model.delayedScanUpdate()

        status = dialog.exec_()

        index = dialog.currentIndex()

        dialog.deleteLater()
        model.deleteLater()

        if status == QDialog.Accepted:
            doc = self.current_document()
            if doc.isModifiedStrict():
                if self.ask_save_changes() == QDialog.Rejected:
                    return QDialog.Rejected

            selected = model.item(index)

            self.load_scheme(str(selected.path()))

        return status

    def tutorial_scheme(self, *args):
        """Browse a collection of tutorial schemes. Returns QDialog.Rejected
        if the user canceled the dialog else loads the selected scheme into
        the canvas and returns QDialog.Accepted.

        """
        tutors = tutorials.tutorials()
        items = [previewmodel.PreviewItem(path=t.abspath()) for t in tutors]
        model = previewmodel.PreviewModel(items=items)
        dialog = previewdialog.PreviewDialog(self)
        title = self.tr("Tutorials")
        dialog.setWindowTitle(title)
        template = ('<h3 style="font-size: 26px">\n'
                    #'<img height="26" src="canvas_icons:Tutorials.svg">\n'
                    '{0}\n'
                    '</h3>')

        dialog.setHeading(template.format(title))
        dialog.setModel(model)

        model.delayedScanUpdate()
        status = dialog.exec_()
        index = dialog.currentIndex()

        dialog.deleteLater()

        if status == QDialog.Accepted:
            doc = self.current_document()
            if doc.isModifiedStrict():
                if self.ask_save_changes() == QDialog.Rejected:
                    return QDialog.Rejected

            selected = model.item(index)

            new_scheme = self.new_scheme_from(str(selected.path()))
            if new_scheme is not None:
                self.set_new_scheme(new_scheme)

        return status

    def welcome_dialog(self):
        """Show a modal welcome dialog for Orange Canvas.
        """

        dialog = welcomedialog.WelcomeDialog(self)
        dialog.setWindowTitle(self.tr("Welcome to Orange Data Mining"))

        def new_scheme():
            if self.new_scheme() == QDialog.Accepted:
                dialog.accept()

        def open_scheme():
            if self.open_scheme() == QDialog.Accepted:
                dialog.accept()

        def open_recent():
            if self.recent_scheme() == QDialog.Accepted:
                dialog.accept()

        def tutorial():
            if self.tutorial_scheme() == QDialog.Accepted:
                dialog.accept()

        new_action = \
            QAction(self.tr("New"), dialog,
                    toolTip=self.tr("Open a new workflow."),
                    triggered=new_scheme,
                    shortcut=QKeySequence.New,
                    icon=canvas_icons("New.svg")
                    )

        open_action = \
            QAction(self.tr("Open"), dialog,
                    objectName="welcome-action-open",
                    toolTip=self.tr("Open a workflow."),
                    triggered=open_scheme,
                    shortcut=QKeySequence.Open,
                    icon=canvas_icons("Open.svg")
                    )

        recent_action = \
            QAction(self.tr("Recent"), dialog,
                    objectName="welcome-recent-action",
                    toolTip=self.tr("Browse and open a recent workflow."),
                    triggered=open_recent,
                    shortcut=QKeySequence(Qt.ControlModifier | \
                                          (Qt.ShiftModifier | Qt.Key_R)),
                    icon=canvas_icons("Recent.svg")
                    )

        tutorials_action = \
            QAction(self.tr("Tutorial"), dialog,
                    objectName="welcome-tutorial-action",
                    toolTip=self.tr("Browse tutorial workflows."),
                    triggered=tutorial,
                    icon=canvas_icons("Tutorials.svg")
                    )

        bottom_row = [self.get_started_action, tutorials_action,
                      self.documentation_action]

        self.new_action.triggered.connect(dialog.accept)
        top_row = [new_action, open_action, recent_action]

        dialog.addRow(top_row, background="light-grass")
        dialog.addRow(bottom_row, background="light-orange")

        settings = QSettings()

        dialog.setShowAtStartup(
            settings.value("startup/show-welcome-screen", True, type=bool)
        )

        status = dialog.exec_()

        settings.setValue("startup/show-welcome-screen",
                          dialog.showAtStartup())

        dialog.deleteLater()

        return status

    def scheme_properties_dialog(self):
        """Return an empty `SchemeInfo` dialog instance.
        """
        settings = QSettings()
        value_key = "schemeinfo/show-at-new-scheme"

        dialog = SchemeInfoDialog(self)

        dialog.setWindowTitle(self.tr("Workflow Info"))
        dialog.setFixedSize(725, 450)

        dialog.setShowAtNewScheme(
            settings.value(value_key, True, type=bool)
        )

        return dialog

    def show_scheme_properties(self):
        """Show current scheme properties.
        """
        settings = QSettings()
        value_key = "schemeinfo/show-at-new-scheme"

        current_doc = self.current_document()
        scheme = current_doc.scheme()
        dlg = self.scheme_properties_dialog()
        dlg.setAutoCommit(False)
        dlg.setScheme(scheme)
        status = dlg.exec_()

        if status == QDialog.Accepted:
            editor = dlg.editor
            stack = current_doc.undoStack()
            stack.beginMacro(self.tr("Change Info"))
            current_doc.setTitle(editor.title())
            current_doc.setDescription(editor.description())
            stack.endMacro()

            # Store the check state.
            settings.setValue(value_key, dlg.showAtNewScheme())
        return status

    def show_scheme_properties_for(self, scheme, window_title=None):
        """Show scheme properties for `scheme` with `window_title (if None
        a default 'Scheme Info' title will be used.

        """
        settings = QSettings()
        value_key = "schemeinfo/show-at-new-scheme"

        dialog = self.scheme_properties_dialog()

        if window_title is not None:
            dialog.setWindowTitle(window_title)

        dialog.setScheme(scheme)

        status = dialog.exec_()
        if status == QDialog.Accepted:
            # Store the check state.
            settings.setValue(value_key, dialog.showAtNewScheme())

        dialog.deleteLater()

        return status

    def set_signal_freeze(self, freeze):
        scheme = self.current_document().scheme()
        manager = scheme.signal_manager
        if freeze:
            manager.pause()
        else:
            manager.resume()

    def remove_selected(self):
        """Remove current scheme selection.
        """
        self.current_document().removeSelected()

    def quit(self):
        """Quit the application.
        """
        if QApplication.activePopupWidget():
            # On OSX the actions in the global menu bar are triggered
            # even if an popup widget is running it's own event loop
            # (in exec_)
            log.debug("Ignoring a quit shortcut during an active "
                      "popup dialog.")
        else:
            self.close()

    def select_all(self):
        self.current_document().selectAll()

    def open_widget(self):
        """Open/raise selected widget's GUI.
        """
        self.current_document().openSelected()

    def rename_widget(self):
        """Rename the current focused widget.
        """
        doc = self.current_document()
        nodes = doc.selectedNodes()
        if len(nodes) == 1:
            doc.editNodeTitle(nodes[0])

    def open_canvas_settings(self):
        """Open canvas settings/preferences dialog
        """
        dlg = UserSettingsDialog(self)
        dlg.setWindowTitle(self.tr("Preferences"))
        dlg.show()
        status = dlg.exec_()
        if status == 0:
            self.__update_from_settings()

    def open_addons(self):
        from .addons import AddonManagerDialog
        dlg = AddonManagerDialog(self, windowTitle=self.tr("Add-ons"))
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        return dlg.exec_()

    def reset_widget_settings(self):
        res = message_question(
            "Clear all widget settings on next restart",
            title="Clear settings",
            informative_text=(
                "A restart of the application is necessary " +
                "for the changes to take effect"),
            buttons=QMessageBox.Ok | QMessageBox.Cancel,
            default_button=QMessageBox.Ok,
            parent=self
        )
        if res == QMessageBox.Ok:
            # Touch a finely crafted file inside the settings directory.
            # The existence of this file is checked by the canvas main
            # function and is deleted there.
            fname = os.path.join(config.widget_settings_dir(),
                                 "DELETE_ON_START")
            os.makedirs(config.widget_settings_dir(), exist_ok=True)
            with open(fname, "a"):
                pass

            if not self.close():
                message_information(
                    "Settings will still be reset at next application start",
                    parent=self)

    def show_output_view(self):
        """Show a window with application output.
        """
        self.output_dock.show()

    def show_report_view(self):
        doc = self.current_document()
        scheme = doc.scheme()
        scheme.show_report_view()

    def output_view(self):
        """Return the output text widget.
        """
        return self.output_dock.widget()

    def open_about(self):
        """Open the about dialog.
        """
        dlg = AboutDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.exec_()

    def add_recent_scheme(self, title, path):
        """Add an entry (`title`, `path`) to the list of recent schemes.
        """
        if not path:
            # No associated persistent path so we can't do anything.
            return

        if not title:
            title = os.path.basename(path)

        filename = os.path.abspath(os.path.realpath(path))
        filename = os.path.normpath(filename)

        actions_by_filename = {}
        for action in self.recent_scheme_action_group.actions():
            path = str(action.data())
            actions_by_filename[path] = action

        if filename in actions_by_filename:
            # Remove the title/filename (so it can be reinserted)
            recent_index = index(self.recent_schemes, filename,
                                 key=operator.itemgetter(1))
            self.recent_schemes.pop(recent_index)

            action = actions_by_filename[filename]
            self.recent_menu.removeAction(action)
            self.recent_scheme_action_group.removeAction(action)
            action.setText(title or self.tr("untitled"))
        else:
            action = QAction(title or self.tr("untitled"), self,
                             toolTip=filename)
            action.setData(filename)

        # Find the separator action in the menu (after 'Browse Recent')
        recent_actions = self.recent_menu.actions()
        begin_index = index(recent_actions, self.recent_menu_begin)
        action_before = recent_actions[begin_index + 1]

        self.recent_menu.insertAction(action_before, action)
        self.recent_scheme_action_group.addAction(action)
        self.recent_schemes.insert(0, (title, filename))

        if len(self.recent_schemes) > max(self.num_recent_schemes, 1):
            title, filename = self.recent_schemes.pop(-1)
            action = actions_by_filename[filename]
            self.recent_menu.removeAction(action)
            self.recent_scheme_action_group.removeAction(action)

        config.save_recent_scheme_list(self.recent_schemes)

    def clear_recent_schemes(self):
        """Clear list of recent schemes
        """
        actions = list(self.recent_menu.actions())

        # Exclude permanent actions (Browse Recent, separators, Clear List)
        actions_to_remove = [action for action in actions \
                             if str(action.data())]

        for action in actions_to_remove:
            self.recent_menu.removeAction(action)
            self.recent_scheme_action_group.removeAction(action)

        self.recent_schemes = []
        config.save_recent_scheme_list([])

    def _on_recent_scheme_action(self, action):
        """A recent scheme action was triggered by the user
        """
        document = self.current_document()
        if document.isModifiedStrict():
            if self.ask_save_changes() == QDialog.Rejected:
                return

        filename = str(action.data())
        self.load_scheme(filename)

    def _on_dock_location_changed(self, location):
        """Location of the dock_widget has changed, fix the margins
        if necessary.

        """
        self.__update_scheme_margins()

    def set_tool_dock_expanded(self, expanded):
        """
        Set the dock widget expanded state.
        """
        self.dock_widget.setExpanded(expanded)

    def _on_tool_dock_expanded(self, expanded):
        """
        'dock_widget' widget was expanded/collapsed.
        """
        if expanded != self.toggle_tool_dock_expand.isChecked():
            self.toggle_tool_dock_expand.setChecked(expanded)

    def createPopupMenu(self):
        # Override the default context menu popup (we don't want the user to
        # be able to hide the tool dock widget).
        return None

    def closeEvent(self, event):
        """Close the main window.
        """
        document = self.current_document()
        if document.isModifiedStrict():
            if self.ask_save_changes() == QDialog.Rejected:
                # Reject the event
                event.ignore()
                return

        old_scheme = document.scheme()

        # Set an empty scheme to clear the document
        document.setScheme(widgetsscheme.WidgetsScheme())

        QApplication.sendEvent(old_scheme, QEvent(QEvent.Close))

        old_scheme.deleteLater()

        config.save_config()

        geometry = self.saveGeometry()
        state = self.saveState(version=self.SETTINGS_VERSION)
        settings = QSettings()
        settings.beginGroup("mainwindow")
        settings.setValue("geometry", geometry)
        settings.setValue("state", state)
        settings.setValue("canvasdock/expanded",
                          self.dock_widget.expanded())
        settings.setValue("scheme-margins-enabled",
                          self.scheme_margins_enabled)

        settings.setValue("last-scheme-dir", self.last_scheme_dir)
        settings.setValue("widgettoolbox/state",
                          self.widgets_tool_box.saveState())

        settings.setValue("quick-help/visible",
                          self.canvas_tool_dock.quickHelpVisible())

        settings.endGroup()

        event.accept()

        # Close any windows left.
        application = QApplication.instance()
        QTimer.singleShot(0, application.closeAllWindows)

    def showEvent(self, event):
        if self.__first_show:
            settings = QSettings()
            settings.beginGroup("mainwindow")

            # Restore geometry and dock/toolbar state
            state = settings.value("state", QByteArray(), type=QByteArray)
            if state:
                self.restoreState(state, version=self.SETTINGS_VERSION)

            geom_data = settings.value("geometry", QByteArray(),
                                       type=QByteArray)
            if geom_data:
                self.restoreGeometry(geom_data)

            self.__first_show = False

        return QMainWindow.showEvent(self, event)

    def event(self, event):
        if event.type() == QEvent.StatusTip and \
                isinstance(event, QuickHelpTipEvent):
            # Using singleShot to update the text browser.
            # If updating directly the application experiences strange random
            # segfaults (in ~StatusTipEvent in QTextLayout or event just normal
            # event loop), but only when the contents are larger then the
            # QTextBrowser's viewport.
            if event.priority() == QuickHelpTipEvent.Normal:
                QTimer.singleShot(0, partial(self.dock_help.showHelp,
                                             event.html()))
            elif event.priority() == QuickHelpTipEvent.Temporary:
                QTimer.singleShot(0, partial(self.dock_help.showHelp,
                                             event.html(), event.timeout()))
            elif event.priority() == QuickHelpTipEvent.Permanent:
                QTimer.singleShot(0, partial(self.dock_help.showPermanentHelp,
                                             event.html()))

            return True

        elif event.type() == QEvent.WhatsThisClicked:
            ref = event.href()
            url = QUrl(ref)

            if url.scheme() == "help" and url.authority() == "search":
                try:
                    url = self.help.search(url)
                except KeyError:
                    url = None
                    log.info("No help topic found for %r", url)

            if url:
                self.show_help(url)
            else:
                message_information(
                    self.tr("There is no documentation for this widget yet."),
                    parent=self)

            return True

        return QMainWindow.event(self, event)

    def show_help(self, url):
        """
        Show `url` in a help window.
        """
        log.info("Setting help to url: %r", url)
        if self.open_in_external_browser:
            url = QUrl(url)
            if not QDesktopServices.openUrl(url):
                # Try fixing some common problems.
                url = QUrl.fromUserInput(url.toString())
                # 'fromUserInput' includes possible fragment into the path
                # (which prevents it to open local files) so we reparse it
                # again.
                url = QUrl(url.toString())
                QDesktopServices.openUrl(url)
        else:
            self.help_view.load(QUrl(url))
            self.help_dock.show()
            self.help_dock.raise_()

    # Mac OS X
    if sys.platform == "darwin":
        def toggleMaximized(self):
            """Toggle normal/maximized window state.
            """
            if self.isMinimized():
                # Do nothing if window is minimized
                return

            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

        def changeEvent(self, event):
            if event.type() == QEvent.WindowStateChange:
                # Can get 'Qt.WindowNoState' before the widget is fully
                # initialized
                if hasattr(self, "window_state"):
                    # Enable/disable window menu based on minimized state
                    self.window_menu.setEnabled(not self.isMinimized())

            QMainWindow.changeEvent(self, event)

    def sizeHint(self):
        """
        Reimplemented from QMainWindow.sizeHint
        """
        hint = QMainWindow.sizeHint(self)
        return hint.expandedTo(QSize(1024, 720))

    def tr(self, sourceText, disambiguation=None, n=-1):
        """Translate the string.
        """
        return str(QMainWindow.tr(self, sourceText, disambiguation, n))

    def __update_from_settings(self):
        settings = QSettings()
        settings.beginGroup("mainwindow")
        toolbox_floatable = settings.value("toolbox-dock-floatable",
                                           defaultValue=False,
                                           type=bool)

        features = self.dock_widget.features()
        features = updated_flags(features, QDockWidget.DockWidgetFloatable,
                                 toolbox_floatable)
        self.dock_widget.setFeatures(features)

        toolbox_exclusive = settings.value("toolbox-dock-exclusive",
                                           defaultValue=True,
                                           type=bool)
        self.widgets_tool_box.setExclusive(toolbox_exclusive)

        self.num_recent_schemes = settings.value("num-recent-schemes",
                                                 defaultValue=15,
                                                 type=int)

        settings.endGroup()
        settings.beginGroup("quickmenu")

        triggers = 0
        dbl_click = settings.value("trigger-on-double-click",
                                   defaultValue=True,
                                   type=bool)
        if dbl_click:
            triggers |= SchemeEditWidget.DoubleClicked

        right_click = settings.value("trigger-on-right-click",
                                    defaultValue=True,
                                    type=bool)
        if right_click:
            triggers |= SchemeEditWidget.RightClicked

        space_press = settings.value("trigger-on-space-key",
                                     defaultValue=True,
                                     type=bool)
        if space_press:
            triggers |= SchemeEditWidget.SpaceKey

        any_press = settings.value("trigger-on-any-key",
                                   defaultValue=False,
                                   type=bool)
        if any_press:
            triggers |= SchemeEditWidget.AnyKey

        self.scheme_widget.setQuickMenuTriggers(triggers)

        settings.endGroup()
        settings.beginGroup("schemeedit")
        show_channel_names = settings.value("show-channel-names",
                                            defaultValue=True,
                                            type=bool)
        self.scheme_widget.setChannelNamesVisible(show_channel_names)

        node_animations = settings.value("enable-node-animations",
                                         defaultValue=False,
                                         type=bool)
        self.scheme_widget.setNodeAnimationEnabled(node_animations)
        settings.endGroup()

        settings.beginGroup("output")
        stay_on_top = settings.value("stay-on-top", defaultValue=True,
                                     type=bool)
        if stay_on_top:
            self.output_dock.setFloatingWindowFlags(Qt.Tool)
        else:
            self.output_dock.setFloatingWindowFlags(Qt.Window)

        dockable = settings.value("dockable", defaultValue=True,
                                  type=bool)
        if dockable:
            self.output_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        else:
            self.output_dock.setAllowedAreas(Qt.NoDockWidgetArea)

        settings.endGroup()

        settings.beginGroup("help")
        stay_on_top = settings.value("stay-on-top", defaultValue=True,
                                     type=bool)
        if stay_on_top:
            self.help_dock.setFloatingWindowFlags(Qt.Tool)
        else:
            self.help_dock.setFloatingWindowFlags(Qt.Window)

        dockable = settings.value("dockable", defaultValue=False,
                                  type=bool)
        if dockable:
            self.help_dock.setAllowedAreas(Qt.LeftDockWidgetArea | \
                                           Qt.RightDockWidgetArea)
        else:
            self.help_dock.setAllowedAreas(Qt.NoDockWidgetArea)

        self.open_in_external_browser = \
            settings.value("open-in-external-browser", defaultValue=False,
                           type=bool)

        self.use_popover = \
            settings.value("toolbox-dock-use-popover-menu", defaultValue=True,
                           type=bool)


def updated_flags(flags, mask, state):
    if state:
        flags |= mask
    else:
        flags &= ~mask
    return flags


def identity(item):
    return item


def index(sequence, *what, **kwargs):
    """index(sequence, what, [key=None, [predicate=None]])

    Return index of `what` in `sequence`.

    """
    what = what[0]
    key = kwargs.get("key", identity)
    predicate = kwargs.get("predicate", operator.eq)
    for i, item in enumerate(sequence):
        item_key = key(item)
        if predicate(what, item_key):
            return i
    raise ValueError("%r not in sequence" % what)
