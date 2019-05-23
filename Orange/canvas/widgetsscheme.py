"""
Orange Widgets Workflow
=======================

A workflow model for Orange Widgets (OWWidget).

This is a subclass of the :class:`~Scheme`. It is responsible for
the construction and management of OWWidget instances corresponding
to the scheme nodes, as well as delegating the signal propagation to a
companion :class:`WidgetsSignalManager` class.

.. autoclass:: WidgetsScheme
   :bases:


.. autoclass:: WidgetsManager
   :bases:

.. autoclass:: WidgetsSignalManager
  :bases:

"""
import copy
import logging
import traceback
import enum
import types
import warnings

from urllib.parse import urlencode
from weakref import finalize

from typing import Optional, Dict, Any

import sip

from AnyQt.QtWidgets import QWidget, QShortcut, QLabel, QSizePolicy, QAction
from AnyQt.QtGui import QKeySequence, QWhatsThisClickedEvent

from AnyQt.QtCore import Qt, QCoreApplication, QEvent, QByteArray
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from Orange.widgets.widget import OWWidget

from orangecanvas.scheme.signalmanager import (
    SignalManager, compress_signals
)
from orangecanvas.scheme.scheme import Scheme, SchemeNode
from orangecanvas.scheme.widgetmanager import WidgetManager as _WidgetManager
from orangecanvas.scheme.node import UserMessage
from orangecanvas.scheme import WorkflowEvent
from orangecanvas.utils import name_lookup
from orangecanvas.resources import icon_loader


log = logging.getLogger(__name__)


class WidgetsScheme(Scheme):
    """
    A Scheme containing Orange Widgets managed with a `WidgetsSignalManager`
    instance.

    Extends the base `Scheme` class to handle the lifetime
    (creation/deletion, etc.) of `OWWidget` instances corresponding to
    the nodes in the scheme. It also delegates the inter-widget signal
    propagation to an instance of `WidgetsSignalManager`.
    """

    #: Emitted when a report_view is requested for the first time, before a
    #: default instance is created. Clients can connect to this signal to
    #: set a report view (`set_report_view`) to use instead.
    report_view_requested = Signal()

    def __init__(self, parent=None, title=None, description=None, env={},
                 **kwargs):
        super().__init__(parent, title, description, env=env, **kwargs)
        self.widget_manager = WidgetManager()
        self.signal_manager = WidgetsSignalManager(self)

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

    def save_widget_geometry_for_node(self, node):
        # type: (SchemeNode) -> bytes
        """
        Save and return the current geometry and state for node

        Parameters
        ----------
        node : Scheme
        """
        w = self.widget_for_node(node)  # type: OWWidget
        return bytes(w.saveGeometryAndLayoutState())

    def restore_widget_geometry_for_node(self, node, state):
        # type: (SchemeNode, bytes) -> bool
        w = self.widget_for_node(node)
        if w is not None:
            return w.restoreGeometryAndLayoutState(QByteArray(state))
        else:
            return False

    def sync_node_properties(self):
        """
        Sync the widget settings/properties with the SchemeNode.properties.
        Return True if there were any changes in the properties (i.e. if the
        new node.properties differ from the old value) and False otherwise.

        """
        changed = False
        for node in self.nodes:
            settings = self.widget_manager.widget_settings_for_node(node)
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
        from Orange.widgets.report.owreport import OWReport
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


class Item(types.SimpleNamespace):
    def __init__(self, node, widget, state, pending_delete=False):
        # type: (SchemeNode, Optional[OWWidget], int, bool) -> None
        super().__init__()
        self.node = node
        self.widget = widget
        self.state = state
        self.pending_delete = pending_delete


class OWWidgetManager(_WidgetManager):
    """
    OWWidget instance manager class.

    This class handles the lifetime of OWWidget instances in a
    :class:`WidgetsScheme`.

    """
    # #: A new OWWidget was created and added by the manager.
    # widget_for_node_added = Signal(SchemeNode, QWidget)
    #
    # #: An OWWidget was removed, hidden and will be deleted when appropriate.
    # widget_for_node_removed = Signal(SchemeNode, QWidget)

    InputUpdate, BlockingUpdate, ProcessingUpdate, Initializing = ProcessingState

    #: State mask for widgets that cannot be deleted immediately
    #: (see __try_delete)
    DelayDeleteMask = InputUpdate | BlockingUpdate

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__scheme = None
        self.__signal_manager = None

        self.__item_for_node = {}  # type: Dict[SchemeNode, Item]

        # Widgets that were 'removed' from the scheme but were at
        # the time in an input update loop and could not be deleted
        # immediately
        self.__delay_delete = {}  # type: Dict[OWWidget, Item]

        # Tracks the widget in the update loop by the SignalManager
        self.__updating_widget = None
        self.__float_widgets_on_top = False

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
        scheme.runtime_env_changed.connect(self.__on_env_changed)
        scheme.installEventFilter(self)
        super().set_workflow(scheme)

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

    def node_for_widget(self, widget):
        # reimplemented
        node = super().node_for_widget(widget)
        if node is None:
            for item in self.__item_for_node.values():
                if item.widget is widget:
                    assert item.state & ProcessingState.Initializing
                    return item.node
        return node

    def widget_settings_for_node(self, node):
        """
        Return the properties/settings from the widget for node.

        Parameters
        ----------
        node : SchemeNode

        Returns
        -------
        settings : dict
        """
        item = self.__item_for_node.get(node)
        if item is not None and isinstance(item.widget, OWWidget):
            return self.widget_settings(item.widget)
        else:
            return node.properties

    def widget_settings(self, widget):
        # type: (OWWidget) -> Dict[str, Any]
        return widget.settingsHandler.pack_data(widget)

    def create_widget_for_node(self, node):
        widget = self.create_widget_instance(node)
        return widget

    def delete_widget_for_node(self, node, widget):
        # type: (SchemeNode, QWidget) -> None
        """
        Reimplemented.
        """
        assert node not in self.workflow().nodes
        item = self.__item_for_node.pop(node, None)
        if item is not None and isinstance(item.widget, OWWidget):
            assert item.node is node
            if item.state & ProcessingState.Initializing:
                raise RuntimeError(
                    "A widget/node {0} was removed while being initialized. "
                    "This is most likely a result of an explicit "
                    "QApplication.processEvents call from the "
                    "'{1.__module__}.{1.__qualname__}' "
                    "widget's __init__."
                    .format(node.title, type(item.widget))
                )
            # Update the node's stored settings/properties dict before
            # removing the widget.
            # TODO: Update/sync whenever the widget settings change.
            node.properties = self.widget_settings(widget)

            node.title_changed.disconnect(widget.setCaption)
            widget.progressBarValueChanged.disconnect(node.set_progress)

            widget.close()
            # Save settings to user global settings.
            widget.saveSettings()
            # Notify the widget it will be deleted.
            widget.onDeleteWidget()
            # Un befriend the report view
            del widget._Report__report_view

            if log.isEnabledFor(logging.DEBUG):
                finalize(
                    widget, log.debug, "Destroyed namespace for: %s", node.title
                )

            # clear the state ?? (have node.reset()??)
            node.set_progress(0)
            node.set_processing_state(0)
            node.set_status_message("")
            msgs = [copy.copy(m) for m in node.state_messages()]
            for m in msgs:
                m.contents = ""
                node.set_state_message(m)

            self.__delete_item(item)

    def __delete_item(self, item):
        item.node = None
        widget = item.widget
        if item.state & WidgetManager.DelayDeleteMask:
            # If the widget is in an update loop and/or blocking we
            # delay the scheduled deletion until the widget is done.
            # The `__on_{processing,blocking}_state_changed` must call
            # __try_delete when the mask is cleared.
            log.debug("Widget %s removed but still in state :%s. "
                      "Deferring deletion.", widget, item.state)
            self.__delay_delete[widget] = item
        else:
            widget.processingStateChanged.disconnect(
                self.__on_processing_state_changed)
            widget.blockingStateChanged.disconnect(
                self.__on_blocking_state_changed)
            widget.deleteLater()
            item.widget = None

    def __try_delete(self, item):
        if not item.state & WidgetManager.DelayDeleteMask:
            widget = item.widget
            log.debug("Delayed delete for widget %s", widget)
            widget.blockingStateChanged.disconnect(
                self.__on_blocking_state_changed)
            widget.processingStateChanged.disconnect(
                self.__on_processing_state_changed)
            item.widget = None
            widget.deleteLater()
            del self.__delay_delete[widget]

    def create_widget_instance(self, node):
        # type: (SchemeNode) -> OWWidget
        """
        Create a OWWidget instance for the node.
        """
        desc = node.description
        klass = widget = None
        initialized = False
        error = None

        item = Item(node, None, ProcessingState.Initializing)
        self.__item_for_node[node] = item

        # First try to actually retrieve the class.
        try:
            klass = name_lookup(desc.qualified_name)
        except ImportError:
            log.exception("", exc_info=True)
            error = "Could not import {0!r}\n\n{1}".format(
                node.description.qualified_name, traceback.format_exc()
            )
        except Exception:
            log.exception("", exc_info=True)
            error = "An unexpected error during import of {0!r}\n\n{1}".format(
                node.description.qualified_name, traceback.format_exc()
            )

        if klass is None:
            widget = mock_error_owwidget(node, error)
            initialized = True

        if widget is None:
            log.info("WidgetManager: Creating '%s.%s' instance '%s'.",
                     klass.__module__, klass.__name__, node.title)

            signal_manager = self.signal_manager()

            widget = klass.__new__(
                klass,
                None,
                captionTitle=node.title,
                signal_manager=signal_manager,
                stored_settings=node.properties,
                # NOTE: env is a view of the real env and reflects
                # changes to the environment.
                env=self.scheme().runtime_env()
            )
            initialized = False

        if not initialized:
            item.widget = widget
            try:
                widget.__init__()
            except Exception:  # pylint: disable=broad-except
                log.exception("", exc_info=True)
                # # sys.excepthook(*sys.exc_info())
                msg = traceback.format_exc()
                msg = "Could not create {0!r}\n\n{1}".format(
                    node.description.name, msg
                )
                # substitute it with a mock error widget.
                item.widget = widget = mock_error_owwidget(node, msg)
                item.state = 0

        # Clear Initializing flag
        item.state &= ~ProcessingState.Initializing
        # initialize and bind the OWWidget to the node
        node.title_changed.connect(widget.setCaption)
        # Widget's info/warning/error messages.
        self.__initialize_widget_state(node, widget)
        widget.messageActivated.connect(self.__on_widget_state_changed)
        widget.messageDeactivated.connect(self.__on_widget_state_changed)

        # Widget's statusMessage
        node.set_status_message(widget.statusMessage())
        widget.statusMessageChanged.connect(node.set_status_message)

        # Widget's progress bar value state.
        widget.progressBarValueChanged.connect(node.set_progress)

        # OWWidget's processing state (progressBarInit/Finished)
        # and the blocking state. We track these for the WidgetsSignalManager.
        widget.processingStateChanged.connect(
            self.__on_processing_state_changed
        )
        widget.blockingStateChanged.connect(self.__on_blocking_state_changed)

        if widget.isBlocking():
            # A widget can already enter blocking state in __init__
            item.state |= ProcessingState.BlockingUpdate

        if widget.processingState != 0:
            # It can also start processing (initialization of resources, ...)
            item.state |= ProcessingState.ProcessingUpdate
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
        return self.__item_for_node[node].state

    def widget_processing_state(self, widget):
        """
        Return the processing state flags for the widget.

        The state is an bitwise or of `InputUpdate` and `BlockingUpdate`.

        """
        node = self.node_for_widget(widget)
        return self.__item_for_node[node].state

    def set_float_widgets_on_top(self, float_on_top):
        """
        Set `Float Widgets on Top` flag on all widgets.
        """
        self.__float_widgets_on_top = float_on_top
        for item in self.__item_for_node.values():
            if item.widget is not None:
                self.__set_float_on_top_flag(item.widget)

    def save_widget_geometry(self, node, widget):
        # type: (SchemeNode, QWidget) -> bytes
        """
        Save and return the current geometry and state for node
        """
        if isinstance(widget, OWWidget):
            return widget.saveGeometryAndLayoutState()
        else:
            return super().save_widget_geometry(node, widget)

    def restore_widget_geometry(self, node, widget, state):
        # type: (SchemeNode, QWidget, bytes) -> bool
        if isinstance(widget, OWWidget):
            return widget.restoreGeometryAndLayoutState(QByteArray(state))
        else:
            return super().restore_widget_geometry(node, widget)

    def eventFilter(self, receiver, event):
        if event.type() == QEvent.Close and receiver is self.__scheme:
            self.signal_manager().stop()

            # Notify the remaining widget instances (if any).
            for item in list(self.__item_for_node.values()):
                widget = item.widget
                if widget is not None:
                    widget.close()
                    widget.saveSettings()
                    widget.onDeleteWidget()
                    widget.deleteLater()

        return super().eventFilter(receiver, event)

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
        event = WorkflowEvent(WorkflowEvent.ActivateParentRequest)
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
        node = self.node_for_widget(widget)
        if node is not None:
            self.__initialize_widget_state(node, widget)

    @Slot(int)
    def __on_processing_state_changed(self, state):
        """
        A widget processing state has changed (progressBarInit/Finished)
        """
        widget = self.sender()
        item = self.__item_for_widget(widget)
        if item is None:
            warnings.warn(
                "State change for a non-tracked widget {}".format(widget),
                RuntimeWarning
            )
            return

        if state:
            item.state |= ProcessingState.ProcessingUpdate
        else:
            item.state &= ~ProcessingState.ProcessingUpdate

        # propagate the change to the workflow model.
        if item.node is not None:
            self.__update_node_processing_state(item.node)

    def __on_processing_started(self, node):
        """
        Signal manager entered the input update loop for the node.
        """
        assert self.__updating_widget is None, "MUST NOT re-enter"
        assert node in self.__item_for_node, "MUST NOT process non-tracked"
        item = self.__item_for_node[node]
        # Remember the widget instance. The node and the node->widget mapping
        # can be removed between this and __on_processing_finished.
        self.__updating_widget = item.widget
        item.state |= ProcessingState.InputUpdate
        self.__update_node_processing_state(node)

    def __on_processing_finished(self, node):
        """
        Signal manager exited the input update loop for the node.
        """
        widget = self.__updating_widget
        self.__updating_widget = None

        item = self.__item_for_widget(widget)
        item.state &= ~ProcessingState.InputUpdate
        if item.node is not None:
            self.__update_node_processing_state(node)
        if widget in self.__delay_delete:
            self.__try_delete(item)

    @Slot(bool)
    def __on_blocking_state_changed(self, state):
        """
        OWWidget blocking state has changed.
        """
        widget = self.sender()
        item = self.__item_for_widget(widget)
        if item is None:
            warnings.warn(
                "State change for a non-tracked widget {}".format(widget),
                RuntimeWarning,
            )
            return

        if not isinstance(widget, OWWidget):
            return

        if not state:
            # unblocked; schedule an signal update pass.
            self.signal_manager()._update()

        if item is not None:
            if state:
                item.state |= ProcessingState.BlockingUpdate
            else:
                item.state &= ~ProcessingState.BlockingUpdate
        if item.node is not None:
            self.__update_node_processing_state(item.node)
        if item.widget in self.__delay_delete:
            self.__try_delete(item)

    def __item_for_widget(self, widget):
        node = self.node_for_widget(widget)
        if node is not None:
            return self.__item_for_node[node]
        else:
            return self.__delay_delete.get(widget)

    def __update_node_processing_state(self, node):
        """
        Update the `node.processing_state` to reflect the widget state.
        """
        state = self.node_processing_state(node)
        node.set_processing_state(1 if state else 0)

    def __on_env_changed(self, key, newvalue, oldvalue):
        # Notify widgets of a runtime environment change
        for item in self.__item_for_node.values():
            if item.widget is not None:
                item.widget.workflowEnvChanged(key, newvalue, oldvalue)

    def __set_float_on_top_flag(self, widget):
        """Set or unset widget's float on top flag"""
        should_float_on_top = self.__float_widgets_on_top
        float_on_top = bool(widget.windowFlags() & Qt.WindowStaysOnTopHint)

        if float_on_top == should_float_on_top:
            return

        widget_was_visible = widget.isVisible()
        if should_float_on_top:
            widget.setWindowFlags(widget.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            widget.setWindowFlags(widget.windowFlags() & ~Qt.WindowStaysOnTopHint)

        # Changing window flags hid the widget
        if widget_was_visible:
            widget.show()


WidgetManager = OWWidgetManager


def user_message_from_state(message_group):
    return UserMessage(
        severity=message_group.severity,
        message_id="{0.__name__}.{0.__qualname__}".format(type(message_group)),
        contents="<br/>".join(msg.formatted
                              for msg in message_group.active) or None,
        data={"content-type": "text/html"})


class WidgetsSignalManager(SignalManager):
    """
    A signal manager for a WidgetsScheme.
    """
    def __init__(self, scheme, **kwargs):
        super().__init__(scheme, **kwargs)
        scheme.installEventFilter(self)

    def send(self, widget, channelname, value, signal_id):
        # type: (OWWidget, str, Any, Any) -> None
        """
        send method compatible with OWWidget.
        """
        scheme = self.scheme()
        node = scheme.widget_manager.node_for_widget(widget)
        if node is None:
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
        # channel name. This is needed for OWWidget's input
        # handlers with Multiple flag.
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

        Deliver input signals to an OWWidget instance.

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
        Process new signals for the OWWidget.
        """
        if sip.isdeleted(widget):
            log.critical("Widget %r was deleted. Cannot process signals",
                         widget)
            return

        app = QCoreApplication.instance()

        for signal in signals:
            link = signal.link
            value = signal.value
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
                log.exception("Error calling '%s' of '%s'",
                              handler.__name__, node.title)
                raise
            finally:
                app.restoreOverrideCursor()

        app.setOverrideCursor(Qt.WaitCursor)
        try:
            widget.handleNewSignals()
        except Exception:
            log.exception("Error calling 'handleNewSignals()' of '%s'",
                          node.title)
            raise
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
