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
import logging

import sip
from PyQt4.QtGui import QShortcut, QKeySequence, QWhatsThisClickedEvent
from PyQt4.QtCore import Qt, QCoreApplication, QEvent, SIGNAL

from .signalmanager import SignalManager, compress_signals, can_enable_dynamic
from .scheme import Scheme, SchemeNode
from .utils import name_lookup, check_arg, check_type
from ..resources import icon_loader
from ..config import rc

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

        self.widgets = []
        self.widget_for_node = {}
        self.node_for_widget = {}
        self.signal_manager = WidgetsSignalManager(self)

    def add_node(self, node):
        """
        Add a `SchemeNode` instance to the scheme and create/initialize the
        OWBaseWidget instance for it.

        """
        check_arg(node not in self.nodes, "Node already in scheme.")
        check_type(node, SchemeNode)

        # Create the widget before a call to Scheme.add_node in
        # case someone connected to node_added already expects
        # widget_for_node, etc. to be up to date.
        widget = self.create_widget_instance(node)

        Scheme.add_node(self, node)

        self.widgets.append(widget)

    def remove_node(self, node):
        Scheme.remove_node(self, node)
        widget = self.widget_for_node[node]

        self.signal_manager.on_node_removed(node)

        del self.widget_for_node[node]
        del self.node_for_widget[widget]

        # Save settings to user global settings.
        widget.saveSettings()

        # Notify the widget it will be deleted.
        widget.onDeleteWidget()
        # And schedule it for deletion.
        widget.deleteLater()

    def add_link(self, link):
        Scheme.add_link(self, link)
        self.signal_manager.link_added(link)

    def remove_link(self, link):
        Scheme.remove_link(self, link)
        self.signal_manager.link_removed(link)

    def create_widget_instance(self, node):
        """
        Create a OWBaseWidget instance for the node.
        """
        desc = node.description
        klass = name_lookup(desc.qualified_name)

        log.info("Creating %r instance.", klass)
        widget = klass.__new__(
            klass,
            _category=desc.category,
            _settingsFromSchema=node.properties
        )

        # Add the node/widget mapping s before calling __init__
        # Some OWWidgets might already send data in the constructor
        # (should this be forbidden? Raise a warning?)
        self.signal_manager.on_node_added(node)

        self.widget_for_node[node] = widget
        self.node_for_widget[widget] = node

        widget.__init__(None, self.signal_manager)
        widget.setCaption(node.title)
        widget.widgetInfo = desc

        widget.setWindowIcon(
            icon_loader.from_description(desc).get(desc.icon)
        )

        widget.setVisible(node.properties.get("visible", False))

        node.title_changed.connect(widget.setCaption)

        # Bind widgets progress/processing state back to the node's properties
        widget.progressBarValueChanged.connect(node.set_progress)
        widget.processingStateChanged.connect(node.set_processing_state)
        self.connect(widget,
                     SIGNAL("blockingStateChanged(bool)"),
                     self.signal_manager._update)

        # Install a help shortcut on the widget
        help_shortcut = QShortcut(QKeySequence("F1"), widget)
        help_shortcut.activated.connect(self.__on_help_request)
        return widget

    def __on_dynamic_link_enabled_changed(self, link, enabled):
        rev = dict(map(reversed, self.widget_for_node.items()))

        source_node = rev[link.widgetFrom]
        sink_node = rev[link.widgetTo]
        source_channel = source_node.output_channel(link.signalNameFrom)
        sink_channel = sink_node.input_channel(link.signalNameTo)

        links = self.find_links(source_node, source_channel,
                                sink_node, sink_channel)

        if links:
            link = links[0]
            link.set_dynamic_enabled(enabled)

    def close_all_open_widgets(self):
        for widget in self.widget_for_node.values():
            widget.onDeleteWidget()
            widget.close()

    def widget_settings(self):
        """Return a list of dictionaries with widget settings.
        """
        return [widget.settingsHandler.pack_data(widget) for widget in
                (self.widget_for_node[node] for node in self.nodes)]

    def save_widget_settings(self):
        """Save all widget settings to their global settings file.
        """
        for node in self.nodes:
            widget = self.widget_for_node[node]
            widget.settingsHandler.update_class_defaults(widget)

    def sync_node_properties(self):
        """Sync the widget settings/properties with the SchemeNode.properties.
        Return True if there were any changes in the properties (i.e. if the
        new node.properties differ from the old value) and False otherwise.

        .. note:: this should hopefully be removed in the feature, when the
            widget can notify a changed setting property.

        """
        changed = False
        for node in self.nodes:
            widget = self.widget_for_node[node]
            settings = widget.getSettings(alsoContexts=False)
            if settings != node.properties:
                node.properties = settings
                changed = True
        log.debug("Scheme node properties sync (changed: %s)", changed)
        return changed

    def save_to(self, stream):
        self.sync_node_properties()
        Scheme.save_to(self, stream)

    def __on_help_request(self):
        """
        Help shortcut was pressed. We send a `QWhatsThisClickedEvent` and
        hope someone responds to it.

        """
        # Sender is the QShortcut, and parent the OWBaseWidget
        widget = self.sender().parent()
        node = self.node_for_widget.get(widget)
        if node:
            url = "help://search?id={0}".format(node.description.id)
            event = QWhatsThisClickedEvent(url)
            QCoreApplication.sendEvent(self, event)


class WidgetsSignalManager(SignalManager):
    def __init__(self, scheme):
        SignalManager.__init__(self, scheme)

        # We keep a mapping from node->widget after the node/widget has been
        # removed from the scheme until we also process all the outgoing signal
        # updates. The reason is the old OWBaseWidget's MULTI channel protocol
        # where the actual source widget instance is passed to the signal
        # handler, and in the delaeyd update the mapping in `scheme()` is no
        # longer available.
        self._widget_backup = {}

        self.freezing = 0

    def on_node_removed(self, node):
        widget = self.scheme().widget_for_node[node]
        SignalManager.on_node_removed(self, node)

        # Store the node->widget mapping for possible delayed signal id.
        # It will be removes in `process_queued` when all signals
        # originating from this widget are delivered.
        self._widget_backup[node] = widget

    def send(self, widget, channelname, value, id):
        """
        send method compatible with OWBaseWidget.
        """
        scheme = self.scheme()
        node = scheme.node_for_widget[widget]

        try:
            channel = node.output_channel(channelname)
        except ValueError:
            log.error("%r is not valid signal name for %r",
                      channelname, node.description.name)
            return

        SignalManager.send(self, node, channel, value, id)

    def is_blocking(self, node):
        return self.scheme().widget_for_node[node].isBlocking()

    def send_to_node(self, node, signals):
        """
        Implementation of `SignalManager.send_to_node`. Deliver data signals
        to OWBaseWidget instance.

        """
        widget = self.scheme().widget_for_node[node]
        self.process_signals_for_widget(node, widget, signals)

    def compress_signals(self, signals):
        return compress_signals(signals)

    def process_queued(self, max_nodes=None):
        SignalManager.process_queued(self, max_nodes=max_nodes)

        # Remove node->widgets backup mapping no longer needed.
        nodes_removed = set(self._widget_backup.keys())
        sources_remaining = set(signal.link.source_node for
                                signal in self._input_queue)

        nodes_to_remove = nodes_removed - sources_remaining
        for node in nodes_to_remove:
            del self._widget_backup[node]

    def process_signals_for_widget(self, node, widget, signals):
        """
        Process new signals for a OWBaseWidget.
        """
        # This replaces the old OWBaseWidget.processSignals method

        if sip.isdeleted(widget):
            log.critical("Widget %r was deleted. Cannot process signals",
                         widget)
            return

        if widget.processingHandler:
            widget.processingHandler(widget, 1)

        scheme = self.scheme()
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
                source_node = link.source_node
                source_name = link.source_channel.name

                if source_node in scheme.widget_for_node:
                    source_widget = scheme.widget_for_node[source_node]
                else:
                    # Node is no longer in the scheme.
                    source_widget = self._widget_backup[source_node]

                # The old OWBaseWidget.processSignals sends the source widget
                # instance along.
                # TODO: Does any widget actually use it, or could it be
                # removed (replaced with a unique id)?
                args = (value, (source_widget, source_name, signal.id))

            log.debug("Process signals: calling %s.%s (from %s with id:%s)",
                      type(widget).__name__, handler.__name__, link, signal.id)

            app.setOverrideCursor(Qt.WaitCursor)
            try:
                handler(*args)
            except Exception:
                log.exception("Error")
            finally:
                app.restoreOverrideCursor()

        app.setOverrideCursor(Qt.WaitCursor)
        try:
            widget.handleNewSignals()
        except Exception:
            log.exception("Error")
        finally:
            app.restoreOverrideCursor()

        # TODO: Test if async processing works, then remove this
        while widget.isBlocking():
            self.thread().msleep(50)
            app.processEvents()

        if widget.processingHandler:
            widget.processingHandler(self, 0)

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
            source_node = scheme.node_for_widget[widgetFrom]
        if widgetTo is not None:
            sink_node = scheme.node_for_widget[widgetTo]

        candidates = scheme.find_links(source_node=source_node,
                                       sink_node=sink_node)

        def signallink(link):
            """
            Construct SignalLink from an SchemeLink.
            """
            w1 = scheme.widget_for_node[link.source_node]
            w2 = scheme.widget_for_node[link.sink_node]

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

        return SignalManager.event(self, event)


class SignalLink(object):
    """
    Back compatiblity with old orngSignalManager, do not use.
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
