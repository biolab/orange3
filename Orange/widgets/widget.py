import contextlib
import sys
import time
import os
import warnings
import types
from functools import reduce

from PyQt4.QtCore import QByteArray, Qt, pyqtSignal as Signal, pyqtProperty,\
    QEventLoop, QSettings, QUrl
from PyQt4.QtGui import QDialog, QPixmap, QLabel, QVBoxLayout, QSizePolicy, \
    qApp, QFrame, QStatusBar, QHBoxLayout, QStyle, QIcon, QApplication, \
    QShortcut, QKeySequence, QDesktopServices, QSplitter, QSplitterHandle

from Orange.data import FileFormat
from Orange.widgets import settings, gui
from Orange.canvas.registry import description as widget_description
from Orange.canvas.report import Report
from Orange.widgets.gui import ControlledAttributesDict, notify_changed
from Orange.widgets.settings import SettingsHandler
from Orange.widgets.utils import vartype, saveplot, getdeepattr
from .utils.overlay import MessageOverlayWidget


def _asmappingproxy(mapping):
    if isinstance(mapping, types.MappingProxyType):
        return mapping
    else:
        return types.MappingProxyType(mapping)


class WidgetMetaClass(type(QDialog)):
    """Meta class for widgets. If the class definition does not have a
       specific settings handler, the meta class provides a default one
       that does not handle contexts. Then it scans for any attributes
       of class settings.Setting: the setting is stored in the handler and
       the value of the attribute is replaced with the default."""

    #noinspection PyMethodParameters
    def __new__(mcs, name, bases, dict):
        from Orange.canvas.registry.description import (
            input_channel_from_args, output_channel_from_args)

        cls = type.__new__(mcs, name, bases, dict)
        if not cls.name: # not a widget
            return cls

        cls.inputs = [input_channel_from_args(inp) for inp in cls.inputs]
        cls.outputs = [output_channel_from_args(outp) for outp in cls.outputs]

        for inp in cls.inputs:
            if not hasattr(cls, inp.handler):
                raise AttributeError("missing input signal handler '{}' in {}".
                                     format(inp.handler, cls.name))

        # TODO Remove this when all widgets are migrated to Orange 3.0
        if (hasattr(cls, "settingsToWidgetCallback") or
                hasattr(cls, "settingsFromWidgetCallback")):
            raise TypeError("Reimplement settingsToWidgetCallback and "
                            "settingsFromWidgetCallback")

        cls.settingsHandler = SettingsHandler.create(cls, template=cls.settingsHandler)

        return cls


class OWWidget(QDialog, Report, metaclass=WidgetMetaClass):
    # Global widget count
    widget_id = 0

    # Widget Meta Description
    # -----------------------

    #: Widget name (:class:`str`) as presented in the Canvas
    name = None
    id = None
    category = None
    version = None
    #: Short widget description (:class:`str` optional), displayed in
    #: canvas help tooltips.
    description = None
    #: A longer widget description (:class:`str` optional)
    long_description = None
    #: Widget icon path relative to the defining module
    icon = "icons/Unknown.png"
    #: Widget priority used for sorting within a category
    #: (default ``sys.maxsize``).
    priority = sys.maxsize

    author = None
    author_email = None
    maintainer = None
    maintainer_email = None
    help = None
    help_ref = None
    url = None
    keywords = []
    background = None
    replaces = None

    #: A list of published input definitions
    inputs = []
    #: A list of published output definitions
    outputs = []

    # Default widget GUI layout settings
    # ----------------------------------

    #: Should the widget have basic layout
    #: (If this flag is false then the `want_main_area` and
    #: `want_control_area` are ignored).
    want_basic_layout = True
    #: Should the widget construct a `mainArea` (this is a resizable
    #: area to the right of the `controlArea`).
    want_main_area = True
    #: Should the widget construct a `controlArea`.
    want_control_area = True
    #: Widget painted by `Save graph" button
    graph_name = None
    graph_writers = FileFormat.img_writers
    want_status_bar = False

    save_position = True

    #: If false the widget will receive fixed size constraint
    #: (derived from it's layout). Use for widgets which have simple
    #: static size contents.
    resizing_enabled = True

    widgetStateChanged = Signal(str, int, str)
    blockingStateChanged = Signal(bool)
    progressBarValueChanged = Signal(float)
    processingStateChanged = Signal(int)

    settingsHandler = None
    """:type: SettingsHandler"""

    savedWidgetGeometry = settings.Setting(None)

    #: A list of advice messages (:class:`Message`) to display to the user.
    #: When a widget is first shown a message from this list is selected
    #: for display. If a user accepts (clicks 'Ok. Got it') the choice is
    #: recorded and the message is never shown again (closing the message
    #: will not mark it as seen). Messages can be displayed again by pressing
    #: Shift + F1
    #:
    #: :type: list of :class:`Message`
    UserAdviceMessages = []

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, None, cls.get_flags())
        QDialog.__init__(self, None, self.get_flags())

        stored_settings = kwargs.get('stored_settings', None)
        if self.settingsHandler:
            self.settingsHandler.initialize(self, stored_settings)

        self.signalManager = kwargs.get('signal_manager', None)
        self.__env = _asmappingproxy(kwargs.get("env", {}))

        setattr(self, gui.CONTROLLED_ATTRIBUTES, ControlledAttributesDict(self))
        self.__reportData = None

        OWWidget.widget_id += 1
        self.widget_id = OWWidget.widget_id

        if self.name:
            self.setCaption(self.name)

        self.setFocusPolicy(Qt.StrongFocus)

        self.startTime = time.time()    # used in progressbar

        self.widgetState = {"Info": {}, "Warning": {}, "Error": {}}

        self.__blocking = False

        # flag indicating if the widget's position was already restored
        self.__was_restored = False

        self.__progressBarValue = -1
        self.__progressState = 0
        self.__statusMessage = ""

        self.__msgwidget = None
        self.__msgchoice = 0

        if self.want_basic_layout:
            self.insertLayout()

        sc = QShortcut(QKeySequence(Qt.ShiftModifier | Qt.Key_F1), self)
        sc.activated.connect(self.__quicktip)

        return self

    def __init__(self, *args, **kwargs):
        """QDialog __init__ was already called in __new__,
        please do not call it here."""

    def inline_graph_report(self):
        box = gui.widgetBox(self.controlArea, orientation="horizontal")
        box.layout().addWidget(self.graphButton)
        box.layout().addWidget(self.report_button)
#        self.report_button_background.hide()
#        self.graphButtonBackground.hide()

    @classmethod
    def get_flags(cls):
        return (Qt.Window if cls.resizing_enabled
                else Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

    class Splitter(QSplitter):
        def createHandle(self):
            return self.Handle(self.orientation(), self,
                                   cursor=Qt.PointingHandCursor)
        class Handle(QSplitterHandle):
            def mouseReleaseEvent(self, event):
                if event.button() == Qt.LeftButton:
                    splitter = self.splitter()
                    splitter.setSizes([int(splitter.sizes()[0] == 0), 1000])
                super().mouseReleaseEvent(event)
            def mouseMoveEvent(self, event):
                return  # Prevent moving; just show/hide

    # noinspection PyAttributeOutsideInit
    def insertLayout(self):
        def createPixmapWidget(self, parent, iconName):
            w = QLabel(parent)
            parent.layout().addWidget(w)
            w.setFixedSize(16, 16)
            w.hide()
            if os.path.exists(iconName):
                w.setPixmap(QPixmap(iconName))
            return w

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(2, 2, 2, 2)

        self.warning_bar = gui.widgetBox(self, orientation="horizontal",
                                         margin=0, spacing=0)
        self.warning_icon = gui.widgetLabel(self.warning_bar, "")
        self.warning_label = gui.widgetLabel(self.warning_bar, "")
        self.warning_label.setStyleSheet("padding-top: 5px")
        self.warning_bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        gui.rubber(self.warning_bar)
        self.warning_bar.setVisible(False)

        self.want_main_area = self.graph_name is not None or self.want_main_area

        splitter = self.Splitter(Qt.Horizontal, self)
        self.layout().addWidget(splitter)

        if self.want_control_area:
            self.controlArea = gui.widgetBox(splitter,
                                             orientation="vertical",
                                             margin=0)
            splitter.setSizes([1])  # Results in smallest size allowed by policy

            if self.graph_name is not None or hasattr(self, "send_report"):
                leftSide = self.controlArea
                self.controlArea = gui.widgetBox(leftSide, margin=0)
            if self.graph_name is not None:
                self.graphButton = gui.button(leftSide, None, "&Save Graph")
                self.graphButton.clicked.connect(self.save_graph)
                self.graphButton.setAutoDefault(0)
            if hasattr(self, "send_report"):
                self.report_button = gui.button(leftSide, None, "&Report",
                                                callback=self.show_report)
                self.report_button.setAutoDefault(0)

            if self.want_main_area:
                self.controlArea.setSizePolicy(QSizePolicy.Fixed,
                                               QSizePolicy.MinimumExpanding)
            self.controlArea.layout().setContentsMargins(4, 4, 0 if self.want_main_area else 4, 4)
        if self.want_main_area:
            self.mainArea = gui.widgetBox(splitter,
                                          orientation="vertical",
                                          margin=4,
                                          sizePolicy=QSizePolicy(QSizePolicy.Expanding,
                                                                 QSizePolicy.Expanding))
            splitter.setCollapsible(1, False)
            self.mainArea.layout().setContentsMargins(0 if self.want_control_area else 4, 4, 4, 4)

        if self.want_status_bar:
            self.widgetStatusArea = QFrame(self)
            self.statusBarIconArea = QFrame(self)
            self.widgetStatusBar = QStatusBar(self)

            self.layout().addWidget(self.widgetStatusArea)

            self.widgetStatusArea.setLayout(QHBoxLayout(self.widgetStatusArea))
            self.widgetStatusArea.layout().addWidget(self.statusBarIconArea)
            self.widgetStatusArea.layout().addWidget(self.widgetStatusBar)
            self.widgetStatusArea.layout().setContentsMargins(0, 0, 0, 0)
            self.widgetStatusArea.setFrameShape(QFrame.StyledPanel)

            self.statusBarIconArea.setLayout(QHBoxLayout())
            self.widgetStatusBar.setSizeGripEnabled(0)

            self.statusBarIconArea.hide()

            self._warningWidget = createPixmapWidget(
                self.statusBarIconArea,
                gui.resource_filename("icons/triangle-orange.png"))
            self._errorWidget = createPixmapWidget(
                self.statusBarIconArea,
                gui.resource_filename("icons/triangle-red.png"))

        if not self.resizing_enabled:
            self.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

    def save_graph(self):
        graph_obj = getdeepattr(self, self.graph_name, None)
        if graph_obj is None:
            return
        saveplot.save_plot(graph_obj, self.graph_writers)

    def updateStatusBarState(self):
        if not hasattr(self, "widgetStatusArea"):
            return
        if self.widgetState["Warning"] or self.widgetState["Error"]:
            self.widgetStatusArea.show()
        else:
            self.widgetStatusArea.hide()

    def setStatusBarText(self, text, timeout=5000):
        if hasattr(self, "widgetStatusBar"):
            self.widgetStatusBar.showMessage(" " + text, timeout)

    def __restoreWidgetGeometry(self):
        restored = False
        if self.save_position:
            geometry = self.savedWidgetGeometry
            if geometry is not None:
                restored = self.restoreGeometry(QByteArray(geometry))

            if restored and not self.windowState() & \
                    (Qt.WindowMaximized | Qt.WindowFullScreen):
                space = qApp.desktop().availableGeometry(self)
                frame, geometry = self.frameGeometry(), self.geometry()

                #Fix the widget size to fit inside the available space
                width = space.width() - (frame.width() - geometry.width())
                width = min(width, geometry.width())
                height = space.height() - (frame.height() - geometry.height())
                height = min(height, geometry.height())
                self.resize(width, height)

                # Move the widget to the center of available space if it is
                # currently outside it
                if not space.contains(self.frameGeometry()):
                    x = max(0, space.width() / 2 - width / 2)
                    y = max(0, space.height() / 2 - height / 2)

                    self.move(x, y)
        return restored

    def __updateSavedGeometry(self):
        if self.__was_restored and self.isVisible():
            # Update the saved geometry only between explicit show/hide
            # events (i.e. changes initiated by the user not by Qt's default
            # window management).
            self.savedWidgetGeometry = bytes(self.saveGeometry())

    # when widget is resized, save the new width and height
    def resizeEvent(self, ev):
        QDialog.resizeEvent(self, ev)
        # Don't store geometry if the widget is not visible
        # (the widget receives a resizeEvent (with the default sizeHint)
        # before first showEvent and we must not overwrite the the
        # savedGeometry with it)
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()

    def moveEvent(self, ev):
        QDialog.moveEvent(self, ev)
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()

    # set widget state to hidden
    def hideEvent(self, ev):
        if self.save_position:
            self.__updateSavedGeometry()
        QDialog.hideEvent(self, ev)

    def closeEvent(self, ev):
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()
        QDialog.closeEvent(self, ev)

    def showEvent(self, ev):
        QDialog.showEvent(self, ev)
        if self.save_position and not self.__was_restored:
            # Restore saved geometry on show
            self.__restoreWidgetGeometry()
            self.__was_restored = True
        self.__quicktipOnce()

    def wheelEvent(self, event):
        # Silently accept the wheel event. This is to ensure combo boxes
        # and other controls that have focus don't receive this event unless
        # the cursor is over them.
        event.accept()

    def setCaption(self, caption):
        # we have to save caption title in case progressbar will change it
        self.captionTitle = str(caption)
        self.setWindowTitle(caption)

    # put this widget on top of all windows
    def reshow(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def send(self, signalName, value, id=None):
        """
        Send a `value` on the `signalName` widget output.

        An output with `signalName` must be defined in the class ``outputs``
        list.
        """
        if not any(s.name == signalName for s in self.outputs):
            raise ValueError('{} is not a valid output signal for widget {}'.format(
                signalName, self.name))
        if self.signalManager is not None:
            self.signalManager.send(self, signalName, value, id)

    def __setattr__(self, name, value):
        """Set value to members of this instance or any of its members.

        If member is used in a gui control, notify the control about the change.

        name: name of the member, dot is used for nesting ("graph.point.size").
        value: value to set to the member.

        """

        names = name.rsplit(".")
        field_name = names.pop()
        obj = reduce(lambda o, n: getattr(o, n, None), names, self)
        if obj is None:
            raise AttributeError("Cannot set '{}' to {} ".format(name, value))

        if obj is self:
            super().__setattr__(field_name, value)
        else:
            setattr(obj, field_name, value)

        notify_changed(obj, field_name, value)

        if self.settingsHandler:
            self.settingsHandler.fast_save(self, name, value)

    def openContext(self, *a):
        self.settingsHandler.open_context(self, *a)

    def closeContext(self):
        self.settingsHandler.close_context(self)

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass

    def saveSettings(self):
        self.settingsHandler.update_defaults(self)

    def onDeleteWidget(self):
        """
        Invoked by the canvas to notify the widget it has been deleted
        from the workflow.

        If possible, subclasses should gracefully cancel any currently
        executing tasks.
        """
        pass

    def handleNewSignals(self):
        """
        Invoked by the workflow signal propagation manager after all
        signals handlers have been called.

        Reimplement this method in order to coalesce updates from
        multiple updated inputs.
        """
        pass

    # ############################################
    # PROGRESS BAR FUNCTIONS

    def progressBarInit(self, processEvents=QEventLoop.AllEvents):
        """
        Initialize the widget's progress (i.e show and set progress to 0%).

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        self.startTime = time.time()
        self.setWindowTitle(self.captionTitle + " (0% complete)")

        if self.__progressState != 1:
            self.__progressState = 1
            self.processingStateChanged.emit(1)

        self.progressBarSet(0, processEvents)

    def progressBarSet(self, value, processEvents=QEventLoop.AllEvents):
        """
        Set the current progress bar to `value`.

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param float value: Progress value
        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        old = self.__progressBarValue
        self.__progressBarValue = value

        if value > 0:
            if self.__progressState != 1:
                warnings.warn("progressBarSet() called without a "
                              "preceding progressBarInit()",
                              stacklevel=2)
                self.__progressState = 1
                self.processingStateChanged.emit(1)

            usedTime = max(1, time.time() - self.startTime)
            totalTime = (100.0 * usedTime) / float(value)
            remainingTime = max(0, totalTime - usedTime)
            h = int(remainingTime / 3600)
            min = int((remainingTime - h * 3600) / 60)
            sec = int(remainingTime - h * 3600 - min * 60)
            if h > 0:
                text = "%(h)d:%(min)02d:%(sec)02d" % vars()
            else:
                text = "%(min)d:%(sec)02d" % vars()
            self.setWindowTitle(self.captionTitle +
                                " (%(value).2f%% complete, remaining time: %(text)s)" % vars())
        else:
            self.setWindowTitle(self.captionTitle + " (0% complete)")

        if old != value:
            self.progressBarValueChanged.emit(value)

        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    def progressBarValue(self):
        return self.__progressBarValue

    progressBarValue = pyqtProperty(float, fset=progressBarSet,
                                    fget=progressBarValue)

    processingState = pyqtProperty(int, fget=lambda self: self.__progressState)

    def progressBarAdvance(self, value, processEvents=QEventLoop.AllEvents):
        self.progressBarSet(self.progressBarValue + value, processEvents)

    def progressBarFinished(self, processEvents=QEventLoop.AllEvents):
        """
        Stop the widget's progress (i.e hide the progress bar).

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        self.setWindowTitle(self.captionTitle)
        if self.__progressState != 0:
            self.__progressState = 0
            self.processingStateChanged.emit(0)

        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    @contextlib.contextmanager
    def progressBar(self, iterations=0):
        """
        Context manager for progress bar.

        Using it ensures that the progress bar is removed at the end without
        needing the `finally` blocks.

        Usage:

            with self.progressBar(20) as progress:
                ...
                progress.advance()

        or

            with self.progressBar() as progress:
                ...
                progress.advance(0.15)

        or

            with self.progressBar():
                ...
                self.progressBarSet(50)

        :param iterations: the number of iterations (optional)
        :type iterations: int
        """
        progress_bar = gui.ProgressBar(self, iterations)
        yield progress_bar
        progress_bar.finish()  # Let us not rely on garbage collector

    #: Widget's status message has changed.
    statusMessageChanged = Signal(str)

    def setStatusMessage(self, text):
        """
        Set widget's status message.

        This is a short status string to be displayed inline next to
        the instantiated widget icon in the canvas.
        """
        if self.__statusMessage != text:
            self.__statusMessage = text
            self.statusMessageChanged.emit(text)

    def statusMessage(self):
        """
        Return the widget's status message.
        """
        return self.__statusMessage

    def keyPressEvent(self, e):
        if (int(e.modifiers()), e.key()) in OWWidget.defaultKeyActions:
            OWWidget.defaultKeyActions[int(e.modifiers()), e.key()](self)
        else:
            QDialog.keyPressEvent(self, e)

    def information(self, id=0, text=None):
        """
        Set/clear a widget information message (for `id`).

        Args:
            id (int or list): The id of the message
            text (str): Text of the message.
        """
        self.setState("Info", id, text)

    def warning(self, id=0, text=""):
        """
        Set/clear a widget warning message (for `id`).

        Args:
            id (int or list): The id of the message
            text (str): Text of the message.
        """
        self.setState("Warning", id, text)

    def error(self, id=0, text=""):
        """
        Set/clear a widget error message (for `id`).

        Args:
            id (int or list): The id of the message
            text (str): Text of the message.
        """
        self.setState("Error", id, text)

    def setState(self, state_type, id, text):
        changed = 0
        if type(id) == list:
            for val in id:
                if val in self.widgetState[state_type]:
                    self.widgetState[state_type].pop(val)
                    changed = 1
        else:
            if type(id) == str:
                text = id
                id = 0
            if not text:
                if id in self.widgetState[state_type]:
                    self.widgetState[state_type].pop(id)
                    changed = 1
            else:
                self.widgetState[state_type][id] = text
                changed = 1

        if changed:
            if type(id) == list:
                for i in id:
                    self.widgetStateChanged.emit(state_type, i, "")
            else:
                self.widgetStateChanged.emit(state_type, id, text or "")

        tooltip_lines = []
        highest_type = None
        for a_type in ("Error", "Warning", "Info"):
            msgs_for_ids = self.widgetState.get(a_type)
            if not msgs_for_ids:
                continue
            msgs_for_ids = list(msgs_for_ids.values())
            if not msgs_for_ids:
                continue
            tooltip_lines += msgs_for_ids
            if highest_type is None:
                highest_type = a_type

        if highest_type is None:
            self.set_warning_bar(None)
        elif len(tooltip_lines) == 1:
            msg = tooltip_lines[0]
            if "\n" in msg:
                self.set_warning_bar(
                    highest_type, msg[:msg.index("\n")] + " (...)", msg)
            else:
                self.set_warning_bar(
                    highest_type, tooltip_lines[0], tooltip_lines[0])
        else:
            self.set_warning_bar(
                highest_type,
                "{} problems during execution".format(len(tooltip_lines)),
                "\n".join(tooltip_lines))

        return changed

    def set_warning_bar(self, state_type, text=None, tooltip=None):
        colors = {"Error": ("#ffc6c6", "black", QStyle.SP_MessageBoxCritical),
                  "Warning": ("#ffffc9", "black", QStyle.SP_MessageBoxWarning),
                  "Info": ("#ceceff", "black", QStyle.SP_MessageBoxInformation)}
        current_height = self.height()
        if state_type is None:
            if not self.warning_bar.isHidden():
                new_height = current_height - self.warning_bar.height()
                self.warning_bar.setVisible(False)
                self.resize(self.width(), new_height)
            return
        background, foreground, icon = colors[state_type]
        style = QApplication.instance().style()
        self.warning_icon.setPixmap(style.standardIcon(icon).pixmap(14, 14))

        self.warning_bar.setStyleSheet(
            "background-color: {}; color: {};"
            "padding: 3px; padding-left: 6px; vertical-align: center".
            format(background, foreground))
        self.warning_label.setText(text)
        self.warning_bar.setToolTip(tooltip)
        if self.warning_bar.isHidden():
            self.warning_bar.setVisible(True)
            new_height = current_height + self.warning_bar.height()
            self.resize(self.width(), new_height)

    def widgetStateToHtml(self, info=True, warning=True, error=True):
        iconpaths = {
            "Info": gui.resource_filename("icons/information.png"),
            "Warning": gui.resource_filename("icons/warning.png"),
            "Error": gui.resource_filename("icons/error.png")
        }
        items = []

        for show, what in [(info, "Info"), (warning, "Warning"),
                           (error, "Error")]:
            if show and self.widgetState[what]:
                items.append('<img src="%s" style="float: left;"> %s' %
                             (iconpaths[what],
                              "\n".join(self.widgetState[what].values())))
        return "<br>".join(items)

    @classmethod
    def getWidgetStateIcons(cls):
        if not hasattr(cls, "_cached__widget_state_icons"):
            info = QPixmap(gui.resource_filename("icons/information.png"))
            warning = QPixmap(gui.resource_filename("icons/warning.png"))
            error = QPixmap(gui.resource_filename("icons/error.png"))
            cls._cached__widget_state_icons = \
                {"Info": info, "Warning": warning, "Error": error}
        return cls._cached__widget_state_icons

    defaultKeyActions = {}

    if sys.platform == "darwin":
        defaultKeyActions = {
            (Qt.ControlModifier, Qt.Key_M):
                lambda self: self.showMaximized
                if self.isMinimized() else self.showMinimized(),
            (Qt.ControlModifier, Qt.Key_W):
                lambda self: self.setVisible(not self.isVisible())}

    def setBlocking(self, state=True):
        """
        Set blocking flag for this widget.

        While this flag is set this widget and all its descendants
        will not receive any new signals from the workflow signal manager.

        This is useful for instance if the widget does it's work in a
        separate thread or schedules processing from the event queue.
        In this case it can set the blocking flag in it's processNewSignals
        method schedule the task and return immediately. After the task
        has completed the widget can clear the flag and send the updated
        outputs.

        .. note::
            Failure to clear this flag will block dependent nodes forever.
        """
        if self.__blocking != state:
            self.__blocking = state
            self.blockingStateChanged.emit(state)

    def isBlocking(self):
        """
        Is this widget blocking signal processing.
        """
        return self.__blocking

    def resetSettings(self):
        self.settingsHandler.reset_settings(self)

    def workflowEnv(self):
        """
        Return (a view to) the workflow runtime environment.

        Returns
        -------
        env : types.MappingProxyType
        """
        return self.__env

    def workflowEnvChanged(self, key, value, oldvalue):
        """
        A workflow environment variable `key` has changed to value.

        Called by the canvas framework to notify widget of a change
        in the workflow runtime environment.

        The default implementation does nothing.
        """
        pass

    def __showMessage(self, message):
        if self.__msgwidget is not None:
            self.__msgwidget.hide()
            self.__msgwidget.deleteLater()
            self.__msgwidget = None

        if message is None:
            return

        buttons = MessageOverlayWidget.Ok | MessageOverlayWidget.Close
        if message.moreurl is not None:
            buttons |= MessageOverlayWidget.Help

        if message.icon is not None:
            icon = message.icon
        else:
            icon = Message.Information

        self.__msgwidget = MessageOverlayWidget(
            parent=self, text=message.text, icon=icon, wordWrap=True,
            standardButtons=buttons)

        b = self.__msgwidget.button(MessageOverlayWidget.Ok)
        b.setText("Ok, got it")

        self.__msgwidget.setStyleSheet("""
            MessageOverlayWidget {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop:0 #666, stop:0.3 #6D6D6D, stop:1 #666)
            }
            MessageOverlayWidget QLabel#text-label {
                color: white;
            }"""
        )

        if message.moreurl is not None:
            helpbutton = self.__msgwidget.button(MessageOverlayWidget.Help)
            helpbutton.setText("Learn more\N{HORIZONTAL ELLIPSIS}")
            self.__msgwidget.helpRequested.connect(
                lambda: QDesktopServices.openUrl(QUrl(message.moreurl)))

        self.__msgwidget.setWidget(self)
        self.__msgwidget.show()

    def __quicktip(self):
        messages = list(self.UserAdviceMessages)
        if messages:
            message = messages[self.__msgchoice % len(messages)]
            self.__msgchoice += 1
            self.__showMessage(message)

    def __quicktipOnce(self):
        filename = os.path.join(settings.widget_settings_dir(),
                               "user-session-state.ini")
        namespace = ("user-message-history/{0.__module__}.{0.__qualname__}"
                     .format(type(self)))
        session_hist = QSettings(filename, QSettings.IniFormat)
        session_hist.beginGroup(namespace)
        messages = self.UserAdviceMessages

        def ispending(msg):
            return not session_hist.value(
                "{}/confirmed".format(msg.persistent_id),
                defaultValue=False, type=bool)
        messages = list(filter(ispending, messages))

        if not messages:
            return

        message = messages[self.__msgchoice % len(messages)]
        self.__msgchoice += 1

        self.__showMessage(message)

        def userconfirmed():
            session_hist = QSettings(filename, QSettings.IniFormat)
            session_hist.beginGroup(namespace)
            session_hist.setValue(
                "{}/confirmed".format(message.persistent_id), True)
            session_hist.sync()

        self.__msgwidget.accepted.connect(userconfirmed)


class Message(object):
    """
    A user message.

    :param str text: Message text
    :param str persistent_id:
        A persistent message id.
    :param icon: Message icon
    :type icon: QIcon or QStyle.StandardPixmap
    :param str moreurl:
        An url to open when a user clicks a 'Learn more' button.

    .. seealso:: :const:`OWWidget.UserAdviceMessages`
    """
    #: QStyle.SP_MessageBox* pixmap enums repeated for easier access
    Question = QStyle.SP_MessageBoxQuestion
    Information = QStyle.SP_MessageBoxInformation
    Warning = QStyle.SP_MessageBoxWarning
    Critical = QStyle.SP_MessageBoxCritical

    def __init__(self, text, persistent_id, icon=None, moreurl=None):
        assert isinstance(text, str)
        assert isinstance(icon, (type(None), QIcon, QStyle.StandardPixmap))
        assert persistent_id is not None
        self.text = text
        self.icon = icon
        self.moreurl = moreurl
        self.persistent_id = persistent_id


#: Input/Output flags.
#: -------------------
#:
#: The input/output is the default for its type.
#: When there are multiple IO signals with the same type the
#: one with the default flag takes precedence when adding a new
#: link in the canvas.
Default = widget_description.Default
NonDefault = widget_description.NonDefault
#: Single input signal (default)
Single = widget_description.Single
#: Multiple outputs can be linked to this signal.
#: Signal handlers with this flag have (object, id: object) -> None signature.
Multiple = widget_description.Multiple
#: Applies to user interaction only.
Explicit = widget_description.Explicit
#: Dynamic output type.
#: Specifies that the instances on the output will in general be
#: subtypes of the declared type and that the output can be connected
#: to any input signal which can accept a subtype of the declared output
#: type.
Dynamic = widget_description.Dynamic

InputSignal = widget_description.InputSignal
OutputSignal = widget_description.OutputSignal


class AttributeList(list):
    pass


class ExampleList(list):
    pass
