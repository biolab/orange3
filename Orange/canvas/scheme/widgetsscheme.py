import logging
from functools import partial

from PyQt4.QtCore import QTimer, SIGNAL

from .. import orngSignalManager
from .scheme import Scheme
from .utils import name_lookup
from ..config import rc
from ..gui.utils import signals_disabled

log = logging.getLogger(__name__)


class WidgetsScheme(Scheme):
    """A Scheme containing Orange Widgets managed with a SignalManager
    instance.

    """
    def __init__(self, parent=None, title=None, description=None):
        Scheme.__init__(self, parent, title, description)

        self.widgets = []
        self.widget_for_node = {}
        self.signal_manager = orngSignalManager.SignalManager()

    def add_node(self, node):
        widget = self.create_widget_instance(node)

        # don't emit the node_added signal until the widget is successfully
        # added to signal manager etc.
        with signals_disabled(self):
            Scheme.add_node(self, node)

        self.widgets.append(widget)

        self.widget_for_node[node] = widget

        self.signal_manager.addWidget(widget)

        self.node_added.emit(node)

    def remove_node(self, node):
        Scheme.remove_node(self, node)
        widget = self.widget_for_node[node]
        self.signal_manager.removeWidget(widget)
        del self.widget_for_node[node]

        # Save settings to user global settings.
        widget.saveSettings()

        # Notify the widget it will be deleted.
        widget.onDeleteWidget()
        # And schedule it for deletion.
        widget.deleteLater()

    def add_link(self, link):
        Scheme.add_link(self, link)
        source_widget = self.widget_for_node[link.source_node]
        sink_widget = self.widget_for_node[link.sink_node]
        source_channel = link.source_channel.name
        sink_channel = link.sink_channel.name
        self.signal_manager.addLink(source_widget, sink_widget, source_channel,
                                    sink_channel, enabled=link.enabled)

        link.enabled_changed.connect(
            partial(self.signal_manager.setLinkEnabled,
                    source_widget, sink_widget)
        )

        QTimer.singleShot(0, self.signal_manager.processNewSignals)

    def remove_link(self, link):
        Scheme.remove_link(self, link)

        source_widget = self.widget_for_node[link.source_node]
        sink_widget = self.widget_for_node[link.sink_node]
        source_channel = link.source_channel.name
        sink_channel = link.sink_channel.name

        self.signal_manager.removeLink(source_widget, sink_widget,
                                       source_channel, sink_channel)

    def create_widget_instance(self, node):
        desc = node.description
        klass = name_lookup(desc.qualified_name)

        log.info("Creating %r instance.", klass)
        widget = klass.__new__(
            klass,
            _category=desc.category,
            _settingsFromSchema=node.properties
        )

        widget.__init__(None, self.signal_manager)
        widget.setCaption(node.title)
        widget.widgetInfo = desc

        widget.setVisible(node.properties.get("visible", False))

        node.title_changed.connect(widget.setCaption)
        # Bind widgets progress/processing state back to the node's properties
        widget.progressBarValueChanged.connect(node.set_progress)
        widget.processingStateChanged.connect(node.set_processing_state)

        # TODO: Change how the signal is emitted in signal manager (should
        # notify the SchemeLink directly).
        widget.connect(
           widget,
           SIGNAL("dynamicLinkEnabledChanged(PyQt_PyObject, bool)"),
           self.__on_dynamic_link_enabled_changed
        )

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
            widget.close()

    def widget_settings(self):
        """Return a list of dictionaries with widget settings.
        """
        return [self.widget_for_node[node].getSettings(alsoContexts=False)
                for node in self.nodes]

    def save_widget_settings(self):
        """Save all widget settings to their global settings file.
        """
        for node in self.nodes:
            widget = self.widget_for_node[node]
            widget.saveSettings()

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
