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

import sip
from PyQt4.QtGui import (
    QShortcut, QKeySequence, QWhatsThisClickedEvent, QWidget
)

from PyQt4.QtCore import Qt, QObject, QCoreApplication, QEvent
from PyQt4.QtCore import pyqtSignal as Signal

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
    def __init__(self, parent=None, title=None, description=None):
        Scheme.__init__(self, parent, title, description)

        self.signal_manager = WidgetsSignalManager(self)
        self.widget_manager = WidgetManager()
        self.widget_manager.set_scheme(self)

    def widget_for_node(self, node):
        """
        Return the OWWidget instance for a `node`
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
            widget = self.widget_for_node(node)
            settings = widget.settingsHandler.pack_data(widget)
            if settings != node.properties:
                node.properties = settings
                changed = True
        log.debug("Scheme node properties sync (changed: %s)", changed)
        return changed

    def show_report_view(self):
        from OWReport import get_instance
        inst = get_instance()
        inst.show()
        inst.raise_()


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

    #: Widget processing state flags:
    #:   * InputUpdate - signal manager is updating/setting the
    #:     widget's inputs
    #:   * BlockingUpdate - widget has entered a blocking state
    #:   * ProcessingUpdate - widget has entered processing state
    InputUpdate, BlockingUpdate, ProcessingUpdate = 1, 2, 4

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.__scheme = None
        self.__signal_manager = None
        self.__widgets = []
        self.__widget_for_node = {}
        self.__node_for_widget = {}

        # Widgets that were 'removed' from the scheme but were at
        # the time in an input update loop and could not be deleted
        # immediately
        self.__delay_delete = set()

        # processing state flags for all nodes (including the ones
        # in __delay_delete).
        self.__widget_processing_state = {}

        # Tracks the widget in the update loop by the SignalManager
        self.__updating_widget = None

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
        return self.__widget_for_node[node]

    def node_for_widget(self, widget):
        """
        Return the SchemeNode instance for the OWWidget.

        Raise a KeyError if the widget does not map to a node in the scheme.
        """
        return self.__node_for_widget[widget]

    def add_widget_for_node(self, node):
        """
        Create a new OWWidget instance for the corresponding scheme node.
        """
        widget = self.create_widget_instance(node)

        self.__widgets.append(widget)
        self.__widget_for_node[node] = widget
        self.__node_for_widget[widget] = node

        self.widget_for_node_added.emit(node, widget)

    def remove_widget_for_node(self, node):
        """
        Remove the OWWidget instance for node.
        """
        widget = self.widget_for_node(node)

        self.__widgets.remove(widget)
        del self.__widget_for_node[node]
        del self.__node_for_widget[widget]

        self.widget_for_node_removed.emit(node, widget)

        self._delete_widget(widget)

    def _delete_widget(self, widget):
        """
        Delete the OWBaseWidget instance.
        """
        widget.close()

        # Save settings to user global settings.
        widget.saveSettings()

        # Notify the widget it will be deleted.
        widget.onDeleteWidget()

        if self.__widget_processing_state[widget] != 0:
            # If the widget is in an update loop and/or blocking we
            # delay the scheduled deletion until the widget is done.
            self.__delay_delete.add(widget)
        else:
            widget.deleteLater()

    def create_widget_instance(self, node):
        """
        Create a OWWidget instance for the node.
        """
        desc = node.description
        klass = name_lookup(desc.qualified_name)

        log.info("Creating %r instance.", klass)
        widget = klass.__new__(
            klass,
            None,
            signal_manager=self.signal_manager(),
            stored_settings=node.properties
        )

        # Init the node/widget mapping and state before calling __init__
        # Some OWWidgets might already send data in the constructor
        # (should this be forbidden? Raise a warning?) triggering the signal
        # manager which would request the widget => node mapping or state
        self.__widget_for_node[node] = widget
        self.__node_for_widget[widget] = node
        self.__widget_processing_state[widget] = 0

        widget.__init__()
        widget.setCaption(node.title)
        widget.widgetInfo = desc

        widget.setWindowIcon(
            icon_loader.from_description(desc).get(desc.icon)
        )

        widget.setVisible(node.properties.get("visible", False))

        node.title_changed.connect(widget.setCaption)

        # Widget's info/warning/error messages.
        widget.widgetStateChanged.connect(self.__on_widget_state_changed)

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

        # Install a help shortcut on the widget
        help_shortcut = QShortcut(QKeySequence("F1"), widget)
        help_shortcut.activated.connect(self.__on_help_request)

        return widget

    def node_processing_state(self, node):
        """
        Return the processing state flags for the node.

        Same as `manager.node_processing_state(manger.widget_for_node(node))`

        """
        widget = self.widget_for_node(node)
        return self.__widget_processing_state[widget]

    def widget_processing_state(self, widget):
        """
        Return the processing state flags for the widget.

        The state is an bitwise or of `InputUpdate` and `BlockingUpdate`.

        """
        return self.__widget_processing_state[widget]

    def eventFilter(self, receiver, event):
        if event.type() == QEvent.Close and receiver is self.__scheme:
            self.signal_manager().stop()

            # Notify the widget instances.
            for widget in list(self.__widget_for_node.values()):
                widget.close()
                widget.saveSettings()
                widget.onDeleteWidget()

            event.accept()
            return True

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
            url = "help://search?id={0}".format(node.description.id)
            event = QWhatsThisClickedEvent(url)
            QCoreApplication.sendEvent(self.scheme(), event)

    def __on_widget_state_changed(self, message_type, message_id,
                                  message_value):
        """
        The OWBaseWidget info/warning/error state has changed.

        message_type is one of "Info", "Warning" or "Error" string depending
        of which method (information, warning, error) was called. message_id
        is the first int argument if supplied, and message_value the message
        text.

        """
        widget = self.sender()
        try:
            node = self.node_for_widget(widget)
        except KeyError:
            pass
        else:
            message_type = str(message_type)
            if message_type == "Info":
                contents = widget.widgetStateToHtml(True, False, False)
                level = UserMessage.Info
            elif message_type == "Warning":
                contents = widget.widgetStateToHtml(False, True, False)
                level = UserMessage.Warning
            elif message_type == "Error":
                contents = widget.widgetStateToHtml(False, False, True)
                level = UserMessage.Error
            else:
                raise ValueError("Invalid message_type: %r" % message_type)

            if not contents:
                contents = None

            message = UserMessage(contents, severity=level,
                                  message_id=message_type,
                                  data={"content-type": "text/html"})
            node.set_state_message(message)

    def __on_processing_state_changed(self, state):
        """
        A widget processing state has changed (progressBarInit/Finished)
        """
        widget = self.sender()
        try:
            node = self.node_for_widget(widget)
        except KeyError:
            return

        if state:
            self.__widget_processing_state[widget] |= self.ProcessingUpdate
        else:
            self.__widget_processing_state[widget] &= ~self.ProcessingUpdate
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
        if self.__widget_processing_state[widget] == 0:
            self.__delay_delete.remove(widget)
            widget.deleteLater()
            del self.__widget_processing_state[widget]


class WidgetsSignalManager(SignalManager):
    """
    A signal manager for a WidgetsScheme.
    """
    def __init__(self, scheme):
        SignalManager.__init__(self, scheme)

        scheme.installEventFilter(self)

        self.freezing = 0

        self.__scheme_deleted = False

        scheme.destroyed.connect(self.__on_scheme_destroyed)
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
            log.debug("Node for %r is not in the scheme.", widget)
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
        return self.scheme().widget_manager.node_processing_state(node) != 0

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

    def scheduleSignalProcessing(self, widget=None):
        """
        Back compatibility with old orngSignalManager.
        """
        self._update()

    def processNewSignals(self, widget=None):
        """
        Back compatibility with old orngSignalManager.

        .. todo:: The old signal manager would update immediately, but
                  this only schedules the update. Is this a problem?

        """
        self._update()

    def addEvent(self, strValue, object=None, eventVerbosity=1):
        """
        Back compatibility with old orngSignalManager module's logging.
        """
        if not isinstance(strValue, str):
            info = str(strValue)
        else:
            info = strValue

        if object is not None:
            info += ". Token type = %s. Value = %s" % \
                    (str(type(object)), str(object)[:100])

        if eventVerbosity > 0:
            log.debug(info)
        else:
            log.info(info)

    def getLinks(self, widgetFrom=None, widgetTo=None,
                 signalNameFrom=None, signalNameTo=None):
        """
        Back compatibility with old orngSignalManager. Some widget look if
        they have any output connections, so this is still needed, but should
        be deprecated in the future.

        """
        scheme = self.scheme()

        source_node = sink_node = None

        if widgetFrom is not None:
            source_node = scheme.node_for_widget(widgetFrom)
        if widgetTo is not None:
            sink_node = scheme.node_for_widget(widgetTo)

        candidates = scheme.find_links(source_node=source_node,
                                       sink_node=sink_node)

        def signallink(link):
            """
            Construct SignalLink from an SchemeLink.
            """
            w1 = scheme.widget_for_node(link.source_node)
            w2 = scheme.widget_for_node(link.sink_node)

            # Input/OutputSignal are reused from description. Interface
            # is almost the same as it was in orngSignalManager
            return SignalLink(w1, link.source_channel,
                              w2, link.sink_channel,
                              link.enabled)

        links = []
        for link in candidates:
            if (signalNameFrom is None or \
                    link.source_channel.name == signalNameFrom) and \
                    (signalNameTo is None or \
                     link.sink_channel.name == signalNameTo):

                links.append(signallink(link))
        return links

    def setFreeze(self, freeze, startWidget=None):
        """
        Freeze/unfreeze signal processing. If freeze >= 1 no signal will be
        processed until freeze is set back to 0.

        """
        self.freezing = max(freeze, 0)
        if freeze > 0:
            log.debug("Freezing signal processing (value:%r set by %r)",
                      freeze, startWidget)
        elif freeze == 0:
            log.debug("Unfreezing signal processing (cleared by %r)",
                      startWidget)

        if self._input_queue:
            self._update()

    def freeze(self, widget=None):
        """
        Return a context manager that freezes the signal processing.
        """
        manager = self

        class freezer(object):
            def __enter__(self):
                self.push()
                return self

            def __exit__(self, *args):
                self.pop()

            def push(self):
                manager.setFreeze(manager.freezing + 1, widget)

            def pop(self):
                manager.setFreeze(manager.freezing - 1, widget)

        return freezer()

    def event(self, event):
        if event.type() == QEvent.UpdateRequest:
            if self.freezing > 0:
                log.debug("received UpdateRequest while signal processing "
                          "is frozen")
                event.setAccepted(False)
                return False

            if self.__scheme_deleted:
                log.debug("Scheme has been/is being deleted. No more "
                          "signals will be delivered to any nodes.")
                event.setAccepted(True)
                return True
        # Retain a reference to the scheme until the 'process_queued' finishes
        # in SignalManager.event.
        scheme = self.scheme()
        return SignalManager.event(self, event)

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
                self.__scheme_deleted = True
                return True

        return SignalManager.eventFilter(self, receiver, event)

    def __on_scheme_destroyed(self, obj):
        self.__scheme_deleted = True


class SignalLink(object):
    """
    Back compatibility with old orngSignalManager, do not use.
    """
    def __init__(self, widgetFrom, outputSignal, widgetTo, inputSignal,
                 enabled=True, dynamic=False):
        self.widgetFrom = widgetFrom
        self.widgetTo = widgetTo

        self.outputSignal = outputSignal
        self.inputSignal = inputSignal

        self.dynamic = dynamic

        self.enabled = enabled

        self.signalNameFrom = self.outputSignal.name
        self.signalNameTo = self.inputSignal.name

    def canEnableDynamic(self, obj):
        """
        Can dynamic signal link be enabled for `obj`?
        """
        return isinstance(obj, name_lookup(self.inputSignal.type))


class SignalWrapper(object):
    """
    Signal (actually slot) wrapper used by OWBaseWidget.connect overload.
    This disables (freezes) the widget's signal manager when slots are
    invoked from GUI signals. Not sure if this is still needed, could instead
    just set the blocking flag on the widget itself.

    """
    def __init__(self, widget, method):
        self.widget = widget
        self.method = method

    def __call__(self, *args):
        manager = self.widget.signalManager
        if manager:
            with manager.freeze(self.method):
                self.method(*args)
        else:
            # Might be running stand alone without a manager.
            self.method(*args)
