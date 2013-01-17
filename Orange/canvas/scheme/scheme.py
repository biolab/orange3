"""
Scheme Workflow

"""

from operator import itemgetter
from collections import deque

import logging

from PyQt4.QtCore import QObject
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property

from .node import SchemeNode
from .link import SchemeLink, compatible_channels
from .annotations import BaseSchemeAnnotation

from .utils import check_arg, check_type

from .errors import (
    SchemeCycleError, IncompatibleChannelTypeError
)

from .readwrite import scheme_to_ows_stream, parse_scheme

from ..registry import WidgetDescription

log = logging.getLogger(__name__)


class Scheme(QObject):
    """An QObject representing the scheme widget workflow
    with annotations, etc.

    """

    node_added = Signal(SchemeNode)
    node_removed = Signal(SchemeNode)

    link_added = Signal(SchemeLink)
    link_removed = Signal(SchemeLink)

    topology_changed = Signal()

    node_state_changed = Signal()
    channel_state_changed = Signal()

    annotation_added = Signal(BaseSchemeAnnotation)
    annotation_removed = Signal(BaseSchemeAnnotation)

    node_property_changed = Signal(SchemeNode, str, object)

    title_changed = Signal(str)
    description_changed = Signal(str)

    def __init__(self, parent=None, title=None, description=None):
        QObject.__init__(self, parent)

        self.__title = title or ""
        "Scheme title (empty string by default)."

        self.__description = description or ""
        "Scheme description (empty string by default)."

        self.__annotations = []
        self.__nodes = []
        self.__links = []

    @property
    def nodes(self):
        return list(self.__nodes)

    @property
    def links(self):
        return list(self.__links)

    @property
    def annotations(self):
        return list(self.__annotations)

    def set_title(self, title):
        if self.__title != title:
            self.__title = title
            self.title_changed.emit(title)

    def title(self):
        return self.__title

    title = Property(str, fget=title, fset=set_title)

    def set_description(self, description):
        if self.__description != description:
            self.__description = description
            self.description_changed.emit(description)

    def description(self):
        return self.__description

    description = Property(str, fget=description, fset=set_description)

    def add_node(self, node):
        """Add a node to the scheme.

        Parameters
        ----------
        node : `SchemeNode`
            Node to add to the scheme.

        """
        check_arg(node not in self.__nodes,
                  "Node already in scheme.")
        check_type(node, SchemeNode)

        self.__nodes.append(node)
        log.info("Added node %r to scheme %r." % (node.title, self.title))
        self.node_added.emit(node)

    def new_node(self, description, title=None, position=None,
                 properties=None):
        """Create a new SchemeNode and add it to the scheme.
        Same as:

            scheme.add_node(SchemeNode(description, title, position,
                                       properties))

        """
        if isinstance(description, WidgetDescription):
            node = SchemeNode(description, title=title, position=position,
                              properties=properties)
        else:
            raise TypeError("Expected %r, got %r." % \
                            (WidgetDescription, type(description)))

        self.add_node(node)
        return node

    def remove_node(self, node):
        """Remove a `node` from the scheme. All links into and out of the node
        are also removed.

        """
        check_arg(node in self.__nodes,
                  "Node is not in the scheme.")

        self.__remove_node_links(node)
        self.__nodes.remove(node)
        log.info("Removed node %r from scheme %r." % (node.title, self.title))
        self.node_removed.emit(node)
        return node

    def __remove_node_links(self, node):
        """Remove all links for node.
        """
        links_in, links_out = [], []
        for link in self.__links:
            if link.source_node is node:
                links_out.append(link)
            elif link.sink_node is node:
                links_in.append(link)

        for link in links_out + links_in:
            self.remove_link(link)

    def add_link(self, link):
        """Add a link to the scheme
        """
        check_type(link, SchemeLink)
        existing = self.find_links(link.source_node, link.source_channel,
                                   link.sink_node, link.sink_channel)
        check_arg(not existing,
                  "Link %r already in the scheme." % link)

        self.check_connect(link)
        self.__links.append(link)

        log.info("Added link %r (%r) -> %r (%r) to scheme %r." % \
                 (link.source_node.title, link.source_channel.name,
                  link.sink_node.title, link.sink_channel.name,
                  self.title)
                 )

        self.link_added.emit(link)

    def new_link(self, source_node, source_channel,
                 sink_node, sink_channel):
        """Crate a new SchemeLink and add it to the scheme.
        Same as:

            scheme.add_link(SchemeLink(source_node, source_channel,
                                       sink_node, sink_channel)

        """
        link = SchemeLink(source_node, source_channel,
                          sink_node, sink_channel)
        self.add_link(link)
        return link

    def remove_link(self, link):
        """Remove a link from the scheme.
        """
        check_arg(link in self.__links,
                  "Link is not in the scheme.")

        self.__links.remove(link)
        log.info("Removed link %r (%r) -> %r (%r) from scheme %r." % \
                 (link.source_node.title, link.source_channel.name,
                  link.sink_node.title, link.sink_channel.name,
                  self.title)
                 )
        self.link_removed.emit(link)

    def check_connect(self, link):
        """Check if the link can be added to the scheme.

        Can raise:
            - `SchemeCycleError` if the link would introduce a cycle
            - `IncompatibleChannelTypeError` if the channel types are not
                compatible

        """
        check_type(link, SchemeLink)
        if self.creates_cycle(link):
            raise SchemeCycleError("Cannot create cycles in the scheme")

        if not self.compatible_channels(link):
            raise IncompatibleChannelTypeError(
                    "Cannot connect %r to %r" \
                    % (link.source_channel, link.sink_channel)
                )

    def creates_cycle(self, link):
        """Would the `link` if added to the scheme introduce a cycle.
        """
        check_type(link, SchemeLink)
        source_node, sink_node = link.source_node, link.sink_node
        upstream = self.upstream_nodes(source_node)
        upstream.add(source_node)
        return sink_node in upstream

    def compatible_channels(self, link):
        """Do the channels in link have compatible types.
        """
        check_type(link, SchemeLink)
        return compatible_channels(link.source_channel, link.sink_channel)

    def can_connect(self, link):
        try:
            self.check_connect(link)
            return True
        except (SchemeCycleError, IncompatibleChannelTypeError):
            return False
        except Exception:
            raise

    def upstream_nodes(self, start_node):
        """Return a set of all nodes upstream from `start_node`.
        """
        visited = set()
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            snodes = [link.source_node for link in self.input_links(node)]
            for source_node in snodes:
                if source_node not in visited:
                    queue.append(source_node)

            visited.add(node)
        visited.remove(start_node)
        return visited

    def downstream_nodes(self, start_node):
        """Return a set of all nodes downstream from `start_node`.
        """
        visited = set()
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            snodes = [link.sink_node for link in self.output_links(node)]
            for source_node in snodes:
                if source_node not in visited:
                    queue.append(source_node)

            visited.add(node)
        visited.remove(start_node)
        return visited

    def is_ancestor(self, node, child):
        """Return True if `node` is an ancestor node of `child` (is upstream
        of the child in the workflow). Both nodes must be in the scheme.

        """
        return child in self.downstream_nodes(node)

    def children(self, node):
        """Return a set of all children of `node`.
        """
        return set(link.sink_node for link in self.output_links(node))

    def parents(self, node):
        """Return a set if all parents of `node`.
        """
        return set(link.source_node for link in self.input_links(node))

    def input_links(self, node):
        """Return all input links connected to the `node`.
        """
        return self.find_links(sink_node=node)

    def output_links(self, node):
        """Return all output links connected to the `node`.
        """
        return self.find_links(source_node=node)

    def find_links(self, source_node=None, source_channel=None,
                   sink_node=None, sink_channel=None):
        # TODO: Speedup - keep index of links by nodes and channels
        result = []
        match = lambda query, value: (query is None or value == query)
        for link in self.__links:
            if match(source_node, link.source_node) and \
                    match(sink_node, link.sink_node) and \
                    match(source_channel, link.source_channel) and \
                    match(sink_channel, link.sink_channel):
                result.append(link)

        return result

    def propose_links(self, source_node, sink_node):
        """Return a list of ordered (`OutputSignal`, `InputSignal`, weight)
        tuples that could be added to the scheme between `source_node` and
        `sink_node`.

        .. note:: This can depend on the links already in the scheme.

        """
        if source_node is sink_node or \
                self.is_ancestor(sink_node, source_node):
            # Cyclic connections are not possible.
            return []

        outputs = source_node.output_channels()
        inputs = sink_node.input_channels()

        # Get existing links to sink channels that are Single.
        links = self.find_links(None, None, sink_node)
        already_connected_sinks = [link.sink_channel for link in links \
                                   if link.sink_channel.single]

        def weight(out_c, in_c):
            if out_c.explicit or in_c.explicit:
                # Zero weight for explicit links
                weight = 0
            else:
                check = [not out_c.dynamic,  # Dynamic signals are last
                         in_c not in already_connected_sinks,
                         bool(in_c.default),
                         bool(out_c.default)
                         ]
                weights = [2 ** i for i in range(len(check), 0, -1)]
                weight = sum([w for w, c in zip(weights, check) if c])
            return weight

        proposed_links = []
        for out_c in outputs:
            for in_c in inputs:
                if compatible_channels(out_c, in_c):
                    proposed_links.append((out_c, in_c, weight(out_c, in_c)))

        return sorted(proposed_links, key=itemgetter(-1), reverse=True)

    def add_annotation(self, annotation):
        """Add an annotation (`BaseSchemeAnnotation`) subclass to the scheme.

        """
        check_arg(annotation not in self.__annotations,
                  "Cannot add the same annotation multiple times.")
        check_type(annotation, BaseSchemeAnnotation)

        self.__annotations.append(annotation)
        self.annotation_added.emit(annotation)

    def remove_annotation(self, annotation):
        check_arg(annotation in self.__annotations,
                  "Annotation is not in the scheme.")
        self.__annotations.remove(annotation)
        self.annotation_removed.emit(annotation)

    def save_to(self, stream, pretty=True):
        """Save the scheme as an xml formated file to `stream`
        """
        if isinstance(stream, str):
            stream = open(stream, "wb")

        scheme_to_ows_stream(self, stream, pretty)

    def load_from(self, stream):
        """Load the scheme from xml formated stream.
        """
        if self.__nodes or self.__links or self.__annotations:
            # TODO: should we clear the scheme and load it.
            raise ValueError("Scheme is not empty.")

        if isinstance(stream, str):
            stream = open(stream, "rb")

        parse_scheme(self, stream)
