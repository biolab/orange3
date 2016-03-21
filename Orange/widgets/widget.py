import sys
import os
import types
from functools import reduce

from AnyQt.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QSizePolicy, QApplication, QStyle,
    QShortcut, QSplitter, QSplitterHandle, QPushButton
)
from AnyQt.QtCore import (
    Qt, QByteArray, QSettings, QUrl, pyqtSignal as Signal
)
from AnyQt.QtGui import QIcon, QKeySequence, QDesktopServices

from Orange.data import FileFormat
from Orange.widgets import settings, gui
from Orange.canvas.registry import description as widget_description
from Orange.canvas.report import Report
from Orange.widgets.gui import ControlledAttributesDict, notify_changed
from Orange.widgets.io import ClipboardFormat
from Orange.widgets.settings import SettingsHandler
from Orange.widgets.utils import saveplot, getdeepattr
from Orange.widgets.utils.progressbar import ProgressBarMixin
from Orange.widgets.utils.messages import \
    WidgetMessagesMixin, UnboundMsg
from .utils.overlay import MessageOverlayWidget

# Msg is imported and renamed, so widgets can import it from this module rather
# than the one with the mixin (Orange.widgets.utils.messages). Assignment is
# used instead of "import ... as", otherwise PyCharm does not suggest import
Msg = UnboundMsg


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
    # pylint: disable=bad-classmethod-argument
    def __new__(mcs, name, bases, kwargs):
        from Orange.canvas.registry.description import (
            input_channel_from_args, output_channel_from_args)

        cls = super().__new__(mcs, name, bases, kwargs)
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


class OWWidget(QDialog, Report, ProgressBarMixin, WidgetMessagesMixin,
               metaclass=WidgetMetaClass):
    """Base widget class"""

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
    #: Widget icon path relative to the defining module
    icon = "icons/Unknown.png"
    #: Widget priority used for sorting within a category
    #: (default ``sys.maxsize``).
    priority = sys.maxsize

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
    #: Orientation of the buttonsArea box; valid only if
    #  `want_control_area` is `True`. Possible values are Qt.Horizontal,
    #  Qt.Vertical and None for no buttons area
    buttons_area_orientation = Qt.Horizontal
    #: Widget painted by `Save graph" button
    graph_name = None
    graph_writers = FileFormat.img_writers

    save_position = True

    #: If false the widget will receive fixed size constraint
    #: (derived from it's layout). Use for widgets which have simple
    #: static size contents.
    resizing_enabled = True

    blockingStateChanged = Signal(bool)
    processingStateChanged = Signal(int)

    # Signals have to be class attributes and cannot be inherited,
    # say from a mixin. This has something to do with the way PyQt binds them
    progressBarValueChanged = Signal(float)
    messageActivated = Signal(Msg)
    messageDeactivated = Signal(Msg)

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

    def __new__(cls, *args, captionTitle=None, **kwargs):
        self = super().__new__(cls, None, cls.get_flags())
        QDialog.__init__(self, None, self.get_flags())
        WidgetMessagesMixin.__init__(self)

        stored_settings = kwargs.get('stored_settings', None)
        if self.settingsHandler:
            self.settingsHandler.initialize(self, stored_settings)

        self.signalManager = kwargs.get('signal_manager', None)
        self.__env = _asmappingproxy(kwargs.get("env", {}))

        setattr(self, gui.CONTROLLED_ATTRIBUTES, ControlledAttributesDict(self))
        self.graphButton = None
        self.report_button = None

        OWWidget.widget_id += 1
        self.widget_id = OWWidget.widget_id

        captionTitle = self.name if captionTitle is None else captionTitle

        # must be set without invoking setCaption
        self.captionTitle = captionTitle
        self.setWindowTitle(captionTitle)

        self.setFocusPolicy(Qt.StrongFocus)

        self.__blocking = False

        # flag indicating if the widget's position was already restored
        self.__was_restored = False

        self.__statusMessage = ""

        self.__msgwidget = None
        self.__msgchoice = 0

        self.left_side = None
        self.controlArea = self.mainArea = self.buttonsArea = None
        self.splitter = None
        if self.want_basic_layout:
            self.set_basic_layout()

        sc = QShortcut(QKeySequence(Qt.ShiftModifier | Qt.Key_F1), self)
        sc.activated.connect(self.__quicktip)

        sc = QShortcut(QKeySequence.Copy, self)
        sc.activated.connect(self.copy_to_clipboard)
        if self.controlArea is not None:
            # Otherwise, the first control has focus
            self.controlArea.setFocus(Qt.ActiveWindowFocusReason)
        return self

    # pylint: disable=super-init-not-called
    def __init__(self, *args, **kwargs):
        """__init__s are called in __new__; don't call them from here"""

    @classmethod
    def get_flags(cls):
        return (Qt.Window if cls.resizing_enabled
                else Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

    class _Splitter(QSplitter):
        def createHandle(self):
            """Create splitter handle"""
            return self._Handle(
                self.orientation(), self, cursor=Qt.PointingHandCursor)

        class _Handle(QSplitterHandle):
            def mouseReleaseEvent(self, event):
                """Resize on left button"""
                if event.button() == Qt.LeftButton:
                    splitter = self.splitter()
                    splitter.setSizes([int(splitter.sizes()[0] == 0), 1000])
                super().mouseReleaseEvent(event)

            def mouseMoveEvent(self, event):
                """Prevent moving; just show/hide"""
                return

    def _insert_splitter(self):
        self.splitter = self._Splitter(Qt.Horizontal, self)
        self.layout().addWidget(self.splitter)

    def _insert_control_area(self):
        self.left_side = gui.vBox(self.splitter, spacing=0)
        self.splitter.setSizes([1])  # Smallest size allowed by policy
        if self.buttons_area_orientation is not None:
            self.controlArea = gui.vBox(self.left_side, addSpace=0)
            self._insert_buttons_area()
        else:
            self.controlArea = self.left_side
        if self.want_main_area:
            self.controlArea.setSizePolicy(
                QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
            m = 0
        else:
            m = 4
        self.controlArea.layout().setContentsMargins(m, m, m, m)

    def _insert_buttons_area(self):
        self.buttonsArea = gui.widgetBox(
            self.left_side, addSpace=0, spacing=9,
            orientation=self.buttons_area_orientation)
        if self.graphButton is not None:
            self.buttonsArea.layout().addWidget(self.graphButton)
        if self.report_button is not None:
            self.buttonsArea.layout().addWidget(self.report_button)

    def _insert_main_area(self):
        self.mainArea = gui.vBox(
            self.splitter, margin=4,
            sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        )
        self.splitter.setCollapsible(1, False)
        self.mainArea.layout().setContentsMargins(
            0 if self.want_control_area else 4, 4, 4, 4)

    def _create_default_buttons(self):
        # These buttons are inserted in buttons_area, if it exists
        # Otherwise it is up to the widget to add them to some layout
        if self.graph_name is not None:
            self.graphButton = QPushButton("&Save Image", autoDefault=False)
            self.graphButton.clicked.connect(self.save_graph)
        if hasattr(self, "send_report"):
            self.report_button = QPushButton("&Report", autoDefault=False)
            self.report_button.clicked.connect(self.show_report)

    def set_basic_layout(self):
        """Provide the basic widget layout

        Which parts are created is regulated by class attributes
        `want_main_area`, `want_control_area` and `buttons_area_orientation`,
        the presence of method `send_report` and attribute `graph_name`.
        """
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(2, 2, 2, 2)
        if not self.resizing_enabled:
            self.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        self.want_main_area = self.want_main_area or self.graph_name
        self._create_default_buttons()
        self.insert_message_bar()
        self._insert_splitter()
        if self.want_control_area:
            self._insert_control_area()
        if self.want_main_area:
            self._insert_main_area()

    def save_graph(self):
        """Save the graph with the name given in class attribute `graph_name`.

        The method is called by the *Save graph* button, which is created
        automatically if the `graph_name` is defined.
        """
        graph_obj = getdeepattr(self, self.graph_name, None)
        if graph_obj is None:
            return
        saveplot.save_plot(graph_obj, self.graph_writers)

    def copy_to_clipboard(self):
        graph_obj = getdeepattr(self, self.graph_name, None)
        if graph_obj is None:
            return
        ClipboardFormat.write_image(None, graph_obj)

    def __restoreWidgetGeometry(self):

        def _fullscreen_to_maximized(geometry):
            """Don't restore windows into full screen mode because it loses
            decorations and can't be de-fullscreened at least on some platforms.
            Use Maximized state insted."""
            w = QWidget(visible=False)
            w.restoreGeometry(QByteArray(geometry))
            if w.isFullScreen():
                w.setWindowState(
                    w.windowState() & ~Qt.WindowFullScreen | Qt.WindowMaximized)
            return w.saveGeometry()

        restored = False
        if self.save_position:
            geometry = self.savedWidgetGeometry
            if geometry is not None:
                geometry = _fullscreen_to_maximized(geometry)
                restored = self.restoreGeometry(geometry)

            if restored and not self.windowState() & \
                    (Qt.WindowMaximized | Qt.WindowFullScreen):
                space = QApplication.desktop().availableGeometry(self)
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
            # Note: This should always be stored as bytes and not QByteArray.
            self.savedWidgetGeometry = bytes(self.saveGeometry())

    # when widget is resized, save the new width and height
    def resizeEvent(self, event):
        """Overloaded to save the geometry (width and height) when the widget
        is resized.
        """
        QDialog.resizeEvent(self, event)
        # Don't store geometry if the widget is not visible
        # (the widget receives a resizeEvent (with the default sizeHint)
        # before first showEvent and we must not overwrite the the
        # savedGeometry with it)
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()

    def moveEvent(self, event):
        """Overloaded to save the geometry when the widget is moved
        """
        QDialog.moveEvent(self, event)
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()

    def hideEvent(self, event):
        """Overloaded to save the geometry when the widget is hidden
        """
        if self.save_position:
            self.__updateSavedGeometry()
        QDialog.hideEvent(self, event)

    def closeEvent(self, event):
        """Overloaded to save the geometry when the widget is closed
        """
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()
        QDialog.closeEvent(self, event)

    def showEvent(self, event):
        """Overloaded to restore the geometry when the widget is shown
        """
        QDialog.showEvent(self, event)
        if self.save_position and not self.__was_restored:
            # Restore saved geometry on show
            self.__restoreWidgetGeometry()
            self.__was_restored = True
        self.__quicktipOnce()

    def wheelEvent(self, event):
        """Silently accept the wheel event.

        This is to ensure combo boxes and other controls that have focus
        don't receive this event unless the cursor is over them.
        """
        event.accept()

    def setCaption(self, caption):
        # save caption title in case progressbar will change it
        self.captionTitle = str(caption)
        self.setWindowTitle(caption)

    def reshow(self):
        """Put the widget on top of all windows
        """
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
        """Open a new context corresponding to the given data.

        The settings handler first checks the stored context for a
        suitable match. If one is found, it becomes the current contexts and
        the widgets settings are initialized accordingly. If no suitable
        context exists, a new context is created and data is copied from
        the widget's settings into the new context.

        Widgets that have context settings must call this method after
        reinitializing the user interface (e.g. combo boxes) with the new
        data.

        The arguments given to this method are passed to the context handler.
        Their type depends upon the handler. For instance,
        `DomainContextHandler` expects `Orange.data.Table` or
        `Orange.data.Domain`.
        """
        self.settingsHandler.open_context(self, *a)

    def closeContext(self):
        """Save the current settings and close the current context.

        Widgets that have context settings must call this method before
        reinitializing the user interface (e.g. combo boxes) with the new
        data.
        """
        self.settingsHandler.close_context(self)

    def retrieveSpecificSettings(self):
        """
        Retrieve data that is not registered as setting.

        This method is called by
        `Orange.widgets.settings.ContextHandler.settings_to_widget`.
        Widgets may define it to retrieve any data that is not stored in widget
        attributes. See :obj:`Orange.widgets.data.owcolor.OWColor` for an
        example.
        """
        pass

    def storeSpecificSettings(self):
        """
        Store data that is not registered as setting.

        This method is called by
        `Orange.widgets.settings.ContextHandler.settings_from_widget`.
        Widgets may define it to store any data that is not stored in widget
        attributes. See :obj:`Orange.widgets.data.owcolor.OWColor` for an
        example.
        """
        pass

    def saveSettings(self):
        """
        Writes widget instance's settings to class defaults. Usually called
        when the widget is deleted.
        """
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
        """Handle default key actions or pass the event to the inherited method
        """
        if (int(e.modifiers()), e.key()) in OWWidget.defaultKeyActions:
            OWWidget.defaultKeyActions[int(e.modifiers()), e.key()](self)
        else:
            QDialog.keyPressEvent(self, e)


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
        """Is this widget blocking signal processing."""
        return self.__blocking

    def resetSettings(self):
        """Reset the widget settings to default"""
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

        btn = self.__msgwidget.button(MessageOverlayWidget.Ok)
        btn.setText("Ok, got it")

        self.__msgwidget.setStyleSheet("""
            MessageOverlayWidget {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop:0 #666, stop:0.3 #6D6D6D, stop:1 #666)
            }
            MessageOverlayWidget QLabel#text-label {
                color: white;
            }""")

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

        def _ispending(msg):
            return not session_hist.value(
                "{}/confirmed".format(msg.persistent_id),
                defaultValue=False, type=bool)
        messages = [msg for msg in messages if _ispending(msg)]

        if not messages:
            return

        message = messages[self.__msgchoice % len(messages)]
        self.__msgchoice += 1

        self.__showMessage(message)

        def _userconfirmed():
            session_hist = QSettings(filename, QSettings.IniFormat)
            session_hist.beginGroup(namespace)
            session_hist.setValue(
                "{}/confirmed".format(message.persistent_id), True)
            session_hist.sync()

        self.__msgwidget.accepted.connect(_userconfirmed)


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
#: Only connected if specifically requested (in a dedicated "Links" dialog)
#: or it is the only possible connection.
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
    """Signal type for lists of attributes (variables)
    """
    pass
