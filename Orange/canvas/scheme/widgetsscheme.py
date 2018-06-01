"""
Widgets Scheme
==============

A Scheme for Orange Widgets Scheme (.ows).

This is a subclass of the general :class:`Scheme`. It is responsible for
the construction and management of OWBaseWidget instances corresponding
to the scheme nodes, as well as delegating the signal propagation to a
companion :class:`WidgetsSignalManager` class.

.. autoclass:: WidgetsScheme
   :bases:

.. autoclass:: WidgetsSignalManager
  :bases:

"""
import sys
import logging
import traceback
import enum
from collections import namedtuple, deque
from urllib.parse import urlencode

import sip

from AnyQt.QtWidgets import QWidget, QShortcut, QLabel, QSizePolicy, QAction, qApp
from AnyQt.QtGui import QKeySequence, QWhatsThisClickedEvent

from AnyQt.QtCore import Qt, QObject, QCoreApplication, QTimer, QEvent
from AnyQt.QtCore import pyqtSignal as Signal

from .signalmanager import SignalManager, compress_signals, can_enable_dynamic
from .scheme import Scheme, SchemeNode
from .node import UserMessage
from ..utils import name_lookup
from ..resources import icon_loader

log = logging.getLogger(__name__)


class WidgetsScheme(Scheme):
    """
    A Scheme containing Orange Widgets managed with a `WidgetsSignalManager`
    instance.

    Extends the base `Scheme` class to handle the lifetime
    (creation/deletion, etc.) of `OWBaseWidget` instances corresponding to
    the nodes in the scheme. It also delegates the interwidget signal
    propagation to an instance of `WidgetsSignalManager`.
    """

    #: Emitted when a report_view is requested for the first time, before a
    #: default instance is created. Clients can connect to this signal to
    #: set a report view (`set_report_view`) to use instead.
    report_view_requested = Signal()

    def __init__(self, parent=None, title=None, description=None, env={}):
        Scheme.__init__(self, parent, title, description, env=env)

        self.signal_manager = WidgetsSignalManager(self)
        self.widget_manager = WidgetManager()

        def onchanged(state):
            # Update widget creation policy based on signal manager's state
            if state == SignalManager.Running:
                self.widget_manager.set_creation_policy(WidgetManager.Normal)
            else:
                self.widget_manager.set_creation_policy(WidgetManager.OnDemand)

        self.signal_manager.stateChanged.connect(onchanged)
        self.widget_manager.set_scheme(self)
        self.__report_view = None  # type: Optional[OWReport]

    def widget_for_node(self, node):
        """
        Return the OWWidget instance for a `node`.
        """
        return self.widget_manager.widget_for_node(node)

    def node_for_widget(self, widget):
        """
        Return the SchemeNode instance for the `widget`.
        """
        return self.widget_manager.node_for_widget(widget)

    def sync_node_properties(self):
        """
        Sync the widget settings/properties with the SchemeNode.properties.
        Return True if there were any changes in the properties (i.e. if the
        new node.properties differ from the old value) and False otherwise.

        """
        changed = False
        for node in self.nodes:
            settings = self.widget_manager.widget_properties(node)
            if settings != node.properties:
                node.properties = settings
                changed = True
        log.debug("Scheme node properties sync (changed: %s)", changed)
        return changed

    def show_report_view(self):
        inst = self.report_view()
        inst.show()
        inst.raise_()

    def has_report(self):
        """
        Does this workflow have an associated report

        Returns
        -------
        has_report: bool
        """
        return self.__report_view is not None

    def report_view(self):
        """
        Return a OWReport instance used by the workflow.

        If a report window has not been set then the `report_view_requested`
        signal is emitted to allow the framework to setup the report window.
        If the report window is still not set after the signal is emitted, a
        new default OWReport instance is constructed and returned.

        Returns
        -------
        report : OWReport
        """
        from Orange.canvas.report.owreport import OWReport
        if self.__report_view is None:
            self.report_view_requested.emit()

        if self.__report_view is None:
            parent = self.parent()
            if isinstance(parent, QWidget):
                window = parent.window()  # type: QWidget
            else:
                window = None

            self.__report_view = OWReport()
            if window is not None:
                self.__report_view.setParent(window, Qt.Window)
        return self.__report_view

    def set_report_view(self, view):
        """
        Set the designated OWReport view for this workflow.

        Parameters
        ----------
        view : Optional[OWReport]
        """
        self.__report_view = view

    def dump_settings(self, node: SchemeNode):
        from Orange.widgets.settings import SettingsPrinter
        widget = self.widget_for_node(node)

        pp = SettingsPrinter(indent=4)
        pp.pprint(widget.settingsHandler.pack_data(widget))

    def event(self, event):
        if event.type() == QEvent.Close and \
                self.__report_view is not None:
            self.__report_view.close()
        return super().event(event)

    def close(self):
        QCoreApplication.sendEvent(self, QEvent(QEvent.Close))


class WidgetManager(QObject):
    """
    OWWidget instance manager class.

    This class handles the lifetime of OWWidget instances in a
    :class:`WidgetsScheme`.

    """
    #: A new OWWidget was created and added by the manager.
    widget_for_node_added = Signal(SchemeNode, QWidget)

    #: An OWWidget was removed, hidden and will be deleted when appropriate.
    widget_for_node_removed = Signal(SchemeNode, QWidget)

    class ProcessingState(enum.IntEnum):
        """Widget processing state flags"""
        #: Signal manager is updating/setting the widget's inputs
        InputUpdate = 1
        #: Widget has entered a blocking state (OWWidget.isBlocking)
        BlockingUpdate = 2
        #: Widget has entered processing state
        ProcessingUpdate = 4
        #: Widget is still in the process of initialization
        Initializing = 8

    InputUpdate, BlockingUpdate, ProcessingUpdate, Initializing = ProcessingState

    #: State mask for widgets that cannot be deleted immediately
    #: (see __try_delete)
    _DelayDeleteMask = InputUpdate | BlockingUpdate

    #: Widget initialization states
    Delayed = namedtuple(
        "Delayed", ["node"])
    PartiallyInitialized = namedtuple(
        "Materializing",
        ["node", "partially_initialized_widget"])
    Materialized = namedtuple(
        "Materialized",
        ["node", "widget"])

    class CreationPolicy(enum.Enum):
        """Widget Creation Policy"""
        #: Widgets are scheduled to be created from the event loop, or when
        #: first accessed with `widget_for_node`
        Normal = "Normal"
        #: Widgets are created immediately when added to the workflow model
        Immediate = "Immediate"
        #: Widgets are created only when first accessed with `widget_for_node`
        OnDemand = "OnDemand"

    Normal, Immediate, OnDemand = CreationPolicy

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.__scheme = None
        self.__signal_manager = None
        self.__widgets = []
        self.__initstate_for_node = {}
        self.__creation_policy = WidgetManager.Normal
        #: a queue of all nodes whose widgets are scheduled for
        #: creation/initialization
        self.__init_queue = deque()  # type: Deque[SchemeNode]
        #: Timer for scheduling widget initialization
        self.__init_timer = QTimer(self, interval=0, singleShot=True)
        self.__init_timer.timeout.connect(self.__create_delayed)

        #: A mapping of SchemeNode -> OWWidget (note: a mapping is only added
        #: after the widget is actually created)
        self.__widget_for_node = {}
        #: a mapping of OWWidget -> SchemeNode
        self.__node_for_widget = {}

        # Widgets that were 'removed' from the scheme but were at
        # the time in an input update loop and could not be deleted
        # immediately
        self.__delay_delete = set()

        #: processing state flags for all widgets (including the ones
        #: in __delay_delete).
        #: Note: widgets which have not yet been created do not have an entry
        self.__widget_processing_state = {}

        # Tracks the widget in the update loop by the SignalManager
        self.__updating_widget = None

        # Widgets float above other windows
        self.__float_widgets_on_top = False
        if hasattr(qApp, "applicationStateChanged"):
            # disables/enables widget floating when app (de)activates
            # available in Qt >= 5.2
            def reapply_float_on_top():
                self.set_float_widgets_on_top(self.__float_widgets_on_top)
            qApp.applicationStateChanged.connect(reapply_float_on_top)

    def set_scheme(self, scheme):
        """
        Set the :class:`WidgetsScheme` instance to manage.
        """
        self.__scheme = scheme
        self.__signal_manager = scheme.findChild(SignalManager)

        self.__signal_manager.processingStarted[SchemeNode].connect(
            self.__on_processing_started
        )
        self.__signal_manager.processingFinished[SchemeNode].connect(
            self.__on_processing_finished
        )
        scheme.node_added.connect(self.add_widget_for_node)
        scheme.node_removed.connect(self.remove_widget_for_node)
        scheme.runtime_env_changed.connect(self.__on_env_changed)
        scheme.installEventFilter(self)

    def scheme(self):
        """
        Return the scheme instance on which this manager is installed.
        """
        return self.__scheme

    def signal_manager(self):
        """
        Return the signal manager in use on the :func:`scheme`.
        """
        return self.__signal_manager

    def widget_for_node(self, node):
        """
        Return the OWWidget instance for the scheme node.
        """
        state = self.__initstate_for_node[node]
        if isinstance(state, WidgetManager.Delayed):
            # Create the widget now if it is still pending
            state = self.__materialize(state)
            return state.widget
        elif isinstance(state, WidgetManager.PartiallyInitialized):
            widget = state.partially_initialized_widget
            log.warning("WidgetManager.widget_for_node: "
                        "Accessing a partially created widget instance. "
                        "This is most likely a result of explicit "
                        "QApplication.processEvents call from the '%s.%s' "
                        "widgets __init__.",
                        type(widget).__module__, type(widget).__name__)
            return widget
        elif isinstance(state, WidgetManager.Materialized):
            return state.widget
        else:
            assert False

    def node_for_widget(self, widget):
        """
        Return the SchemeNode instance for the OWWidget.

        Raise a KeyError if the widget does not map to a node in the scheme.
        """
        return self.__node_for_widget[widget]

    def widget_properties(self, node):
        """
        Return the current widget properties/settings.

        Parameters
        ----------
        node : SchemeNode

        Returns
        -------
        settings : dict
        """
        state = self.__initstate_for_node[node]
        if isinstance(state, WidgetManager.Materialized):
            return state.widget.settingsHandler.pack_data(state.widget)
        else:
            return node.properties

    def set_creation_policy(self, policy):
        """
        Set the widget creation policy

        Parameters
        ----------
        policy : WidgetManager.CreationPolicy
        """
        if self.__creation_policy != policy:
            self.__creation_policy = policy

            if self.__creation_policy == WidgetManager.Immediate:
                self.__init_timer.stop()
                while self.__init_queue:
                    state = self.__init_queue.popleft()
                    self.__materialize(state)
            elif self.__creation_policy == WidgetManager.Normal:
                if not self.__init_timer.isActive() and self.__init_queue:
                    self.__init_timer.start()
            elif self.__creation_policy == WidgetManager.OnDemand:
                self.__init_timer.stop()
            else:
                assert False

    def creation_policy(self):
        """
        Return the current widget creation policy

        Returns
        -------
        policy: WidgetManager.CreationPolicy
        """
        return self.__creation_policy

    def add_widget_for_node(self, node):
        """
        Create a new OWWidget instance for the corresponding scheme node.
        """
        state = WidgetManager.Delayed(node)
        self.__initstate_for_node[node] = state

        if self.__creation_policy == WidgetManager.Immediate:
            self.__initstate_for_node[node] = self.__materialize(state)
        elif self.__creation_policy == WidgetManager.Normal:
            self.__init_queue.append(state)
            if not self.__init_timer.isActive():
                self.__init_timer.start()
        elif self.__creation_policy == WidgetManager.OnDemand:
            self.__init_queue.append(state)

    def __materialize(self, state):
        # Create and initialize an OWWidget for a Delayed
        # widget initialization
        assert isinstance(state, WidgetManager.Delayed)
        if state in self.__init_queue:
            self.__init_queue.remove(state)

        node = state.node

        widget = self.create_widget_instance(node)

        self.__widgets.append(widget)
        self.__widget_for_node[node] = widget
        self.__node_for_widget[widget] = node

        self.__initialize_widget_state(node, widget)

        state = WidgetManager.Materialized(node, widget)
        self.__initstate_for_node[node] = state
        self.widget_for_node_added.emit(node, widget)

        return state

    def remove_widget_for_node(self, node):
        """
        Remove the OWWidget instance for node.
        """
        state = self.__initstate_for_node[node]
        if isinstance(state, WidgetManager.Delayed):
            del self.__initstate_for_node[node]
            self.__init_queue.remove(state)
        elif isinstance(state, WidgetManager.Materialized):
            # Update the node's stored settings/properties dict before
            # removing the widget.
            # TODO: Update/sync whenever the widget settings change.
            node.properties = self._widget_settings(state.widget)
            self.__widgets.remove(state.widget)
            del self.__initstate_for_node[node]
            del self.__widget_for_node[node]
            del self.__node_for_widget[state.widget]
            node.title_changed.disconnect(state.widget.setCaption)
            state.widget.progressBarValueChanged.disconnect(node.set_progress)
            del state.widget._Report__report_view
            self.widget_for_node_removed.emit(node, state.widget)
            self._delete_widget(state.widget)
        elif isinstance(state, WidgetManager.PartiallyInitialized):
            widget = state.partially_initialized_widget
            raise RuntimeError(
                "A widget/node {} was removed while being initialized. "
                "This is most likely a result of an explicit "
                "QApplication.processEvents call from the '{}.{}' "
                "widgets __init__.\n"
                .format(state.node.title, type(widget).__module__,
                        type(widget).__init__))

    def _widget_settings(self, widget):
        return widget.settingsHandler.pack_data(widget)

    def _delete_widget(self, widget):
        """
        Delete the OWBaseWidget instance.
        """
        widget.close()
        # Save settings to user global settings.
        widget.saveSettings()
        # Notify the widget it will be deleted.
        widget.onDeleteWidget()

        state = self.__widget_processing_state[widget]
        if state & WidgetManager._DelayDeleteMask:
            # If the widget is in an update loop and/or blocking we
            # delay the scheduled deletion until the widget is done.
            log.debug("Widget %s removed but still in state :%s. "
                      "Deferring deletion.", widget, state)
            self.__delay_delete.add(widget)
        else:
            widget.deleteLater()
            del self.__widget_processing_state[widget]

    def create_widget_instance(self, node):
        """
        Create a OWWidget instance for the node.
        """
        desc = node.description
        klass = widget = None
        initialized = False
        error = None
        # First try to actually retrieve the class.
        try:
            klass = name_lookup(desc.qualified_name)
        except (ImportError, AttributeError):
            sys.excepthook(*sys.exc_info())
            error = "Could not import {0!r}\n\n{1}".format(
                node.description.qualified_name, traceback.format_exc()
            )
        except Exception:
            sys.excepthook(*sys.exc_info())
            error = "An unexpected error during import of {0!r}\n\n{1}".format(
                node.description.qualified_name, traceback.format_exc()
            )

        if klass is None:
            widget = mock_error_owwidget(node, error)
            initialized = True

        if widget is None:
            log.info("WidgetManager: Creating '%s.%s' instance '%s'.",
                     klass.__module__, klass.__name__, node.title)

            widget = klass.__new__(
                klass,
                None,
                captionTitle=node.title,
                signal_manager=self.signal_manager(),
                stored_settings=node.properties,
                # NOTE: env is a view of the real env and reflects
                # changes to the environment.
                env=self.scheme().runtime_env()
            )
            initialized = False

        # Init the node/widget mapping and state before calling __init__
        # Some OWWidgets might already send data in the constructor
        # (should this be forbidden? Raise a warning?) triggering the signal
        # manager which would request the widget => node mapping or state
        # Furthermore they can (though they REALLY REALLY REALLY should not)
        # explicitly call qApp.processEvents.
        assert node not in self.__widget_for_node
        self.__widget_for_node[node] = widget
        self.__node_for_widget[widget] = node
        self.__widget_processing_state[widget] = WidgetManager.Initializing
        self.__initstate_for_node[node] = \
            WidgetManager.PartiallyInitialized(node, widget)

        if not initialized:
            try:
                widget.__init__()
            except Exception:
                sys.excepthook(*sys.exc_info())
                msg = traceback.format_exc()
                msg = "Could not create {0!r}\n\n{1}".format(
                    node.description.name, msg
                )
                # remove state tracking for widget ...
                del self.__widget_for_node[node]
                del self.__node_for_widget[widget]
                del self.__widget_processing_state[widget]

                # ... and substitute it with a mock error widget.
                widget = mock_error_owwidget(node, msg)
                self.__widget_for_node[node] = widget
                self.__node_for_widget[widget] = node
                self.__widget_processing_state[widget] = 0
                self.__initstate_for_node[node] = \
                    WidgetManager.Materialized(node, widget)

        self.__initstate_for_node[node] = \
            WidgetManager.Materialized(node, widget)
        # Clear Initializing flag
        self.__widget_processing_state[widget] &= ~WidgetManager.Initializing

        node.title_changed.connect(widget.setCaption)

        # Widget's info/warning/error messages.
        widget.messageActivated.connect(self.__on_widget_state_changed)
        widget.messageDeactivated.connect(self.__on_widget_state_changed)

        # Widget's statusTip
        node.set_status_message(widget.statusMessage())
        widget.statusMessageChanged.connect(node.set_status_message)

        # Widget's progress bar value state.
        widget.progressBarValueChanged.connect(node.set_progress)

        # Widget processing state (progressBarInit/Finished)
        # and the blocking state.
        widget.processingStateChanged.connect(
            self.__on_processing_state_changed
        )
        widget.blockingStateChanged.connect(self.__on_blocking_state_changed)

        if widget.isBlocking():
            # A widget can already enter blocking state in __init__
            self.__widget_processing_state[widget] |= self.BlockingUpdate

        if widget.processingState != 0:
            # It can also start processing (initialization of resources, ...)
            self.__widget_processing_state[widget] |= self.ProcessingUpdate
            node.set_processing_state(1)
            node.set_progress(widget.progressBarValue)

        # Install a help shortcut on the widget
        help_action = widget.findChild(QAction, "action-help")
        if help_action is not None:
            help_action.setEnabled(True)
            help_action.setVisible(True)
            help_action.triggered.connect(self.__on_help_request)

        # Up shortcut (activate/open parent)
        up_shortcut = QShortcut(
            QKeySequence(Qt.ControlModifier + Qt.Key_Up), widget)
        up_shortcut.activated.connect(self.__on_activate_parent)

        # Call setters only after initialization.
        widget.setWindowIcon(
            icon_loader.from_description(desc).get(desc.icon)
        )
        widget.setCaption(node.title)
        # befriend class Report
        widget._Report__report_view = self.scheme().report_view

        self.__set_float_on_top_flag(widget)

        # Schedule an update with the signal manager, due to the cleared
        # implicit Initializing flag
        self.signal_manager()._update()

        return widget

    def node_processing_state(self, node):
        """
        Return the processing state flags for the node.

        Same as `manager.widget_processing_state(manger.widget_for_node(node))`

        """
        state = self.__initstate_for_node[node]
        if isinstance(state, WidgetManager.Materialized):
            return self.__widget_processing_state[state.widget]
        elif isinstance(state, WidgetManager.PartiallyInitialized):
            return self.__widget_processing_state[state.partially_initialized_widget]
        else:
            return WidgetManager.Initializing

    def widget_processing_state(self, widget):
        """
        Return the processing state flags for the widget.

        The state is an bitwise or of `InputUpdate` and `BlockingUpdate`.

        """
        return self.__widget_processing_state[widget]

    def set_float_widgets_on_top(self, float_on_top):
        """
        Set `Float Widgets on Top` flag on all widgets.
        """
        self.__float_widgets_on_top = float_on_top

        for widget in self.__widget_for_node.values():
            self.__set_float_on_top_flag(widget)

    def __create_delayed(self):
        if self.__init_queue:
            state = self.__init_queue.popleft()
            node = state.node
            self.__initstate_for_node[node] = self.__materialize(state)

        if self.__creation_policy == WidgetManager.Normal and \
                self.__init_queue:
            # restart the timer if pending widgets still in the queue
            self.__init_timer.start()

    def eventFilter(self, receiver, event):
        if event.type() == QEvent.Close and receiver is self.__scheme:
            self.signal_manager().stop()

            # Notify the widget instances.
            for widget in list(self.__widget_for_node.values()):
                widget.close()
                widget.saveSettings()
                widget.onDeleteWidget()
                widget.deleteLater()

        return QObject.eventFilter(self, receiver, event)

    def __on_help_request(self):
        """
        Help shortcut was pressed. We send a `QWhatsThisClickedEvent` to
        the scheme and hope someone responds to it.

        """
        # Sender is the QShortcut, and parent the OWBaseWidget
        widget = self.sender().parent()
        try:
            node = self.node_for_widget(widget)
        except KeyError:
            pass
        else:
            qualified_name = node.description.qualified_name
            help_url = "help://search?" + urlencode({"id": qualified_name})
            event = QWhatsThisClickedEvent(help_url)
            QCoreApplication.sendEvent(self.scheme(), event)

    def __on_activate_parent(self):
        """
        Activate parent shortcut was pressed.
        """
        event = ActivateParentEvent()
        QCoreApplication.sendEvent(self.scheme(), event)

    def __initialize_widget_state(self, node, widget):
        """
        Initialize the tracked info/warning/error message state.
        """
        for message_group in widget.message_groups:
            message = user_message_from_state(message_group)
            if message:
                node.set_state_message(message)

    def __on_widget_state_changed(self, msg):
        """
        The OWBaseWidget info/warning/error state has changed.
        """
        widget = msg.group.widget
        try:
            node = self.node_for_widget(widget)
        except KeyError:
            pass
        else:
            self.__initialize_widget_state(node, widget)

    def __on_processing_state_changed(self, state):
        """
        A widget processing state has changed (progressBarInit/Finished)
        """
        widget = self.sender()

        if state:
            self.__widget_processing_state[widget] |= self.ProcessingUpdate
        else:
            self.__widget_processing_state[widget] &= ~self.ProcessingUpdate

        # propagate the change to the workflow model.
        try:
            # we can still track widget state after it was removed from the
            # workflow model (`__delay_delete`)
            node = self.node_for_widget(widget)
        except KeyError:
            pass
        else:
            self.__update_node_processing_state(node)

    def __on_processing_started(self, node):
        """
        Signal manager entered the input update loop for the node.
        """
        widget = self.widget_for_node(node)
        # Remember the widget instance. The node and the node->widget mapping
        # can be removed between this and __on_processing_finished.
        self.__updating_widget = widget
        self.__widget_processing_state[widget] |= self.InputUpdate
        self.__update_node_processing_state(node)

    def __on_processing_finished(self, node):
        """
        Signal manager exited the input update loop for the node.
        """
        widget = self.__updating_widget
        self.__widget_processing_state[widget] &= ~self.InputUpdate

        if widget in self.__node_for_widget:
            self.__update_node_processing_state(node)
        elif widget in self.__delay_delete:
            self.__try_delete(widget)
        else:
            raise ValueError("%r is not managed" % widget)

        self.__updating_widget = None

    def __on_blocking_state_changed(self, state):
        """
        OWWidget blocking state has changed.
        """
        if not state:
            # schedule an update pass.
            self.signal_manager()._update()

        widget = self.sender()
        if state:
            self.__widget_processing_state[widget] |= self.BlockingUpdate
        else:
            self.__widget_processing_state[widget] &= ~self.BlockingUpdate

        if widget in self.__node_for_widget:
            node = self.node_for_widget(widget)
            self.__update_node_processing_state(node)

        elif widget in self.__delay_delete:
            self.__try_delete(widget)

    def __update_node_processing_state(self, node):
        """
        Update the `node.processing_state` to reflect the widget state.
        """
        state = self.node_processing_state(node)
        node.set_processing_state(1 if state else 0)

    def __try_delete(self, widget):
        if not (self.__widget_processing_state[widget]
                & WidgetManager._DelayDeleteMask):
            log.debug("Delayed delete for widget %s", widget)
            self.__delay_delete.remove(widget)
            del self.__widget_processing_state[widget]
            widget.blockingStateChanged.disconnect(
                self.__on_blocking_state_changed)
            widget.processingStateChanged.disconnect(
                self.__on_processing_state_changed)
            widget.deleteLater()

    def __on_env_changed(self, key, newvalue, oldvalue):
        # Notify widgets of a runtime environment change
        for widget in self.__widget_for_node.values():
            widget.workflowEnvChanged(key, newvalue, oldvalue)

    def __set_float_on_top_flag(self, widget):
        """Set or unset widget's float on top flag"""
        should_float_on_top = self.__float_widgets_on_top
        if hasattr(qApp, "applicationState"):
            # only float on top when the application is active
            # available in Qt >= 5.2
            should_float_on_top &= qApp.applicationState() == Qt.ApplicationActive
        float_on_top = widget.windowFlags() & Qt.WindowStaysOnTopHint

        if float_on_top == should_float_on_top:
            return

        widget_was_visible = widget.isVisible()
        if should_float_on_top:
            widget.setWindowFlags(Qt.WindowStaysOnTopHint)
        else:
            widget.setWindowFlags(widget.windowFlags() & ~Qt.WindowStaysOnTopHint)

        # Changing window flags hid the widget
        if widget_was_visible:
            widget.show()


def user_message_from_state(message_group):
    return UserMessage(
        severity=message_group.severity,
        message_id=message_group,
        contents="<br/>".join(msg.formatted
                              for msg in message_group.active) or None,
        data={"content-type": "text/html"})


class WidgetsSignalManager(SignalManager):
    """
    A signal manager for a WidgetsScheme.
    """
    def __init__(self, scheme):
        SignalManager.__init__(self, scheme)

        scheme.installEventFilter(self)
        scheme.node_added.connect(self.on_node_added)
        scheme.node_removed.connect(self.on_node_removed)
        scheme.link_added.connect(self.link_added)
        scheme.link_removed.connect(self.link_removed)

    def send(self, widget, channelname, value, signal_id):
        """
        send method compatible with OWBaseWidget.
        """
        scheme = self.scheme()
        try:
            node = scheme.node_for_widget(widget)
        except KeyError:
            # The Node/Widget was already removed from the scheme.
            log.debug("Node for '%s' (%s.%s) is not in the scheme.",
                      widget.captionTitle,
                      type(widget).__module__, type(widget).__name__)
            return

        try:
            channel = node.output_channel(channelname)
        except ValueError:
            log.error("%r is not valid signal name for %r",
                      channelname, node.description.name)
            return

        # Expand the signal_id with the unique widget id and the
        # channel name. This is needed for OWBaseWidget's input
        # handlers (Multiple flag).
        signal_id = (widget.widget_id, channelname, signal_id)

        SignalManager.send(self, node, channel, value, signal_id)

    def is_blocking(self, node):
        """Reimplemented from `SignalManager`"""
        mask = (WidgetManager.InputUpdate |
                WidgetManager.BlockingUpdate |
                WidgetManager.Initializing)
        return self.scheme().widget_manager.node_processing_state(node) & mask

    def send_to_node(self, node, signals):
        """
        Implementation of `SignalManager.send_to_node`.

        Deliver input signals to an OWBaseWidget instance.

        """
        widget = self.scheme().widget_for_node(node)
        self.process_signals_for_widget(node, widget, signals)

    def compress_signals(self, signals):
        """
        Reimplemented from :func:`SignalManager.compress_signals`.
        """
        return compress_signals(signals)

    def process_signals_for_widget(self, node, widget, signals):
        """
        Process new signals for the OWBaseWidget.
        """
        # This replaces the old OWBaseWidget.processSignals method

        if sip.isdeleted(widget):
            log.critical("Widget %r was deleted. Cannot process signals",
                         widget)
            return

        app = QCoreApplication.instance()

        for signal in signals:
            link = signal.link
            value = signal.value

            # Check and update the dynamic link state
            if link.is_dynamic():
                link.dynamic_enabled = can_enable_dynamic(link, value)
                if not link.dynamic_enabled:
                    # Send None instead
                    value = None

            handler = link.sink_channel.handler
            if handler.startswith("self."):
                handler = handler.split(".", 1)[1]

            handler = getattr(widget, handler)

            if link.sink_channel.single:
                args = (value,)
            else:
                args = (value, signal.id)

            log.debug("Process signals: calling %s.%s (from %s with id:%s)",
                      type(widget).__name__, handler.__name__, link, signal.id)

            app.setOverrideCursor(Qt.WaitCursor)
            try:
                handler(*args)
            except Exception:
                sys.excepthook(*sys.exc_info())
                log.exception("Error calling '%s' of '%s'",
                              handler.__name__, node.title)
            finally:
                app.restoreOverrideCursor()

        app.setOverrideCursor(Qt.WaitCursor)
        try:
            widget.handleNewSignals()
        except Exception:
            sys.excepthook(*sys.exc_info())
            log.exception("Error calling 'handleNewSignals()' of '%s'",
                          node.title)
        finally:
            app.restoreOverrideCursor()

    def eventFilter(self, receiver, event):
        if event.type() == QEvent.DeferredDelete and receiver is self.scheme():
            try:
                state = self.runtime_state()
            except AttributeError:
                # If the scheme (which is a parent of this object) is
                # already being deleted the SignalManager can also be in
                # the process of destruction (noticeable by its __dict__
                # being empty). There is nothing really to do in this
                # case.
                state = None

            if state == SignalManager.Processing:
                log.info("Deferring a 'DeferredDelete' event for the Scheme "
                         "instance until SignalManager exits the current "
                         "update loop.")
                event.setAccepted(False)
                self.processingFinished.connect(self.scheme().deleteLater)
                self.stop()
                return True

        return SignalManager.eventFilter(self, receiver, event)


class ActivateParentEvent(QEvent):
    ActivateParent = QEvent.registerEventType()

    def __init__(self):
        QEvent.__init__(self, self.ActivateParent)


def mock_error_owwidget(node, message):
    """
    Create a mock OWWidget instance for `node`.

    Parameters
    ----------
    node : SchemeNode
    message : str
    """
    from Orange.widgets import widget, settings

    class DummyOWWidget(widget.OWWidget):
        """
        Dummy OWWidget used to report import/init errors in the canvas.
        """
        name = "Placeholder"

        # Fake settings handler that preserves the settings
        class DummySettingsHandler(settings.SettingsHandler):
            def pack_data(self, widget):
                return getattr(widget, "_settings", {})

            def initialize(self, widget, data=None):
                widget._settings = data
                settings.SettingsHandler.initialize(self, widget, None)

            # specifically disable persistent global defaults
            def write_defaults(self):
                pass

            def read_defaults(self):
                pass

        settingsHandler = DummySettingsHandler()

        want_main_area = False

        def __init__(self, parent=None):
            super().__init__(parent)
            self.errorLabel = QLabel(
                textInteractionFlags=Qt.TextSelectableByMouse,
                wordWrap=True,
            )
            self.errorLabel.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            self.controlArea.layout().addWidget(self.errorLabel)

        def setErrorMessage(self, message):
            self.errorLabel.setText(message)
            self.error(message)

    widget = DummyOWWidget()
    widget._settings = node.properties

    for link in node.description.inputs:
        handler = link.handler
        if handler.startswith("self."):
            _, handler = handler.split(".", 1)

        setattr(widget, handler, lambda *args: None)

    widget.setErrorMessage(message)
    return widget
