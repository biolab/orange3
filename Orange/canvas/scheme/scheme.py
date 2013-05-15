"""
===============
Scheme Workflow
===============

The :class:`Scheme` class defines a DAG (Directed Acyclic Graph) workflow.

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
    SchemeCycleError, IncompatibleChannelTypeError, SinkChannelError,
    DuplicatedLinkError
)

from .readwrite import scheme_to_ows_stream, parse_scheme

from ..registry import WidgetDescription, InputSignal, OutputSignal

log = logging.getLogger(__name__)


class Scheme(QObject):
    """
    An :class:`QObject` subclass representing the scheme widget workflow
    with annotations.

    Parameters
    ----------
    parent : :class:`QObject`
        A parent QObject item (default `None`).
    title : str
        The scheme title.
    description : str
        A longer description of the scheme.


    Attributes
    ----------
    nodes : list of :class:`.SchemeNode`
        A list of all the nodes in the scheme.

    links : list of :class:`.SchemeLink`
        A list of all links in the scheme.

    annotations : list of :class:`BaseSchemeAnnotation`
        A list of all the annotations in the scheme.

    """

    # Signal emitted when a `node` is added to the scheme.
    node_added = Signal(SchemeNode)

    # Signal emitted when a `node` is removed from the scheme.
    node_removed = Signal(SchemeNode)

    # Signal emitted when a `link` is added to the scheme.
    link_added = Signal(SchemeLink)

    # Signal emitted when a `link` is removed from the scheme.
    link_removed = Signal(SchemeLink)

    # Signal emitted when a `annotation` is added to the scheme.
    annotation_added = Signal(BaseSchemeAnnotation)

    # Signal emitted when a `annotation` is removed from the scheme.
    annotation_removed = Signal(BaseSchemeAnnotation)

    # Signal emitted when the title of scheme changes.
    title_changed = Signal(str)

    # Signal emitted when the description of scheme changes.
    description_changed = Signal(str)

    node_state_changed = Signal()
    channel_state_changed = Signal()
    topology_changed = Signal()

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
        """
        A list of all nodes (:class:`.SchemeNode`) currently in the scheme.
        """
        return list(self.__nodes)

    @property
    def links(self):
        """
        A list of all links (:class:`.SchemeLink`) currently in the scheme.
        """
        return list(self.__links)

    @property
    def annotations(self):
        """
        A list of all annotations (:class:`.BaseSchemeAnnotation`) in the
        scheme.

        """
        return list(self.__annotations)

    def set_title(self, title):
        """
        Set the scheme title text.
        """
        if self.__title != title:
            self.__title = title
            self.title_changed.emit(title)

    def title(self):
        """
        The title (human readable string) of the scheme.
        """
        return self.__title

    title = Property(str, fget=title, fset=set_title)

    def set_description(self, description):
        """
        Set the scheme description text.
        """
        if self.__description != description:
            self.__description = description
            self.description_changed.emit(description)

    def description(self):
        """
        Scheme description text.
        """
        return self.__description

    description = Property(str, fget=description, fset=set_description)

    def add_node(self, node):
        """
        Add a node to the scheme. An error is raised if the node is
        already in the scheme.

        Parameters
        ----------
        node : :class:`.SchemeNode`
            Node instance to add to the scheme.

        """
        check_arg(node not in self.__nodes,
                  "Node already in scheme.")
        check_type(node, SchemeNode)

        self.__nodes.append(node)
        log.info("Added node %r to scheme %r." % (node.title, self.title))
        self.node_added.emit(node)

    def new_node(self, description, title=None, position=None,
                 properties=None):
        """
        Create a new :class:`.SchemeNode` and add it to the scheme.

        Same as::

            scheme.add_node(SchemeNode(description, title, position,
                                       properties))

        Parameters
        ----------
        description : :class:`WidgetDescription`
            The new node's description.
        title : str, optional
            Optional new nodes title. By default `description.name` is used.
        position : `(x, y)` tuple of floats, optional
            Optional position in a 2D space.
        properties : dict, optional
            A dictionary of optional extra properties.

        See also
        --------
        .SchemeNode, Scheme.add_node

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
        """
        Remove a `node` from the scheme. All links into and out of the
        `node` are also removed. If the node in not in the scheme an error
        is raised.

        Parameters
        ----------
        node : :class:`.SchemeNode`
            Node instance to remove.

        """
        check_arg(node in self.__nodes,
                  "Node is not in the scheme.")

        self.__remove_node_links(node)
        self.__nodes.remove(node)
        log.info("Removed node %r from scheme %r." % (node.title, self.title))
        self.node_removed.emit(node)
        return node

    def __remove_node_links(self, node):
        """
        Remove all links for node.
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
        """
        Add a `link` to the scheme.

        Parameters
        ----------
        link : :class:`.SchemeLink`
            An initialized link instance to add to the scheme.

        """
        check_type(link, SchemeLink)

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
        """
        Create a new :class:`.SchemeLink` from arguments and add it to
        the scheme. The new link is returned.

        Parameters
        ----------
        source_node : :class:`.SchemeNode`
            Source node of the new link.
        source_channel : :class:`.OutputSignal`
            Source channel of the new node. The instance must be from
            ``source_node.output_channels()``
        sink_node : :class:`.SchemeNode`
            Sink node of the new link.
        sink_channel : :class:`.InputSignal`
            Sink channel of the new node. The instance must be from
            ``sink_node.input_channels()``

        See also
        --------
        .SchemeLink, Scheme.add_link

        """
        link = SchemeLink(source_node, source_channel,
                          sink_node, sink_channel)
        self.add_link(link)
        return link

    def remove_link(self, link):
        """
        Remove a link from the scheme.

        Parameters
        ----------
        link : :class:`.SchemeLink`
            Link instance to remove.

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
        """
        Check if the `link` can be added to the scheme and raise an
        appropriate exception.

        Can raise:
            - :class:`TypeError` if `link` is not an instance of
              :class:`.SchemeLink`
            - :class:`.SchemeCycleError` if the `link` would introduce a cycle
            - :class:`.IncompatibleChannelTypeError` if the channel types are
              not compatible
            - :class:`.SinkChannelError` if a sink channel has a `Single` flag
              specification and the channel is already connected.
            - :class:`.DuplicatedLinkError` if a `link` duplicates an already
              present link.

        """
        check_type(link, SchemeLink)

        if self.creates_cycle(link):
            raise SchemeCycleError("Cannot create cycles in the scheme")

        if not self.compatible_channels(link):
            raise IncompatibleChannelTypeError(
                    "Cannot connect %r to %r." \
                    % (link.source_channel.type, link.sink_channel.type)
                )

        links = self.find_links(source_node=link.source_node,
                                source_channel=link.source_channel,
                                sink_node=link.sink_node,
                                sink_channel=link.sink_channel)

        if links:
            raise DuplicatedLinkError(
                    "A link from %r (%r) -> %r (%r) already exists" \
                    % (link.source_node.title, link.source_channel.name,
                       link.sink_node.title, link.sink_channel.name)
                )

        if link.sink_channel.single:
            links = self.find_links(sink_node=link.sink_node,
                                    sink_channel=link.sink_channel)
            if links:
                raise SinkChannelError(
                        "%r is already connected." % link.sink_channel.name
                    )

    def creates_cycle(self, link):
        """
        Return `True` if `link` would introduce a cycle in the scheme.

        Parameters
        ----------
        link : :class:`.SchemeLink`

        """
        check_type(link, SchemeLink)
        source_node, sink_node = link.source_node, link.sink_node
        upstream = self.upstream_nodes(source_node)
        upstream.add(source_node)
        return sink_node in upstream

    def compatible_channels(self, link):
        """
        Return `True` if the channels in `link` have compatible types.

        Parameters
        ----------
        link : :class:`.SchemeLink`

        """
        check_type(link, SchemeLink)
        return compatible_channels(link.source_channel, link.sink_channel)

    def can_connect(self, link):
        """
        Return `True` if `link` can be added to the scheme.

        See also
        --------
        Scheme.check_connect

        """
        check_type(link, SchemeLink)
        try:
            self.check_connect(link)
            return True
        except (SchemeCycleError, IncompatibleChannelTypeError,
                SinkChannelError, DuplicatedLinkError):
            return False

    def upstream_nodes(self, start_node):
        """
        Return a set of all nodes upstream from `start_node` (i.e.
        all ancestor nodes).

        Parameters
        ----------
        start_node : :class:`.SchemeNode`

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
        """
        Return a set of all nodes downstream from `start_node`.

        Parameters
        ----------
        start_node : :class:`.SchemeNode`

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
        """
        Return True if `node` is an ancestor node of `child` (is upstream
        of the child in the workflow). Both nodes must be in the scheme.

        Parameters
        ----------
        node : :class:`.SchemeNode`
        child : :class:`.SchemeNode`

        """
        return child in self.downstream_nodes(node)

    def children(self, node):
        """
        Return a set of all children of `node`.
        """
        return set(link.sink_node for link in self.output_links(node))

    def parents(self, node):
        """
        Return a set of all parents of `node`.
        """
        return set(link.source_node for link in self.input_links(node))

    def input_links(self, node):
        """
        Return a list of all input links (:class:`.SchemeLink`) connected
        to the `node` instance.

        """
        return self.find_links(sink_node=node)

    def output_links(self, node):
        """
        Return a list of all output links (:class:`.SchemeLink`) connected
        to the `node` instance.

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
        """
        Return a list of ordered (:class:`OutputSignal`,
        :class:`InputSignal`, weight) tuples that could be added to
        the scheme between `source_node` and `sink_node`.

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
        """
        Add an annotation (:class:`BaseSchemeAnnotation` subclass) instance
        to the scheme.

        """
        check_arg(annotation not in self.__annotations,
                  "Cannot add the same annotation multiple times.")
        check_type(annotation, BaseSchemeAnnotation)

        self.__annotations.append(annotation)
        self.annotation_added.emit(annotation)

    def remove_annotation(self, annotation):
        """
        Remove the `annotation` instance from the scheme.
        """
        check_arg(annotation in self.__annotations,
                  "Annotation is not in the scheme.")
        self.__annotations.remove(annotation)
        self.annotation_removed.emit(annotation)

    def clear(self):
        """
        Remove all nodes, links, and annotation items from the scheme.
        """
        def is_terminal(node):
            return not bool(self.find_links(source_node=node))

        while self.nodes:
            terminal_nodes = list(filter(is_terminal, self.nodes))
            for node in terminal_nodes:
                self.remove_node(node)

        for annotation in self.annotations:
            self.remove_annotation(annotation)

        assert(not (self.nodes or self.links or self.annotations))

    def save_to(self, stream, pretty=True, pickle_fallback=False):
        """
        Save the scheme as an xml formated file to `stream`

        See also
        --------
        .scheme_to_ows_stream

        """
        if isinstance(stream, str):
            stream = open(stream, "wb")

        scheme_to_ows_stream(self, stream, pretty,
                             pickle_fallback=pickle_fallback)

    def load_from(self, stream):
        """
        Load the scheme from xml formated stream.
        """
        if self.__nodes or self.__links or self.__annotations:
            # TODO: should we clear the scheme and load it.
            raise ValueError("Scheme is not empty.")

        if isinstance(stream, str):
            stream = open(stream, "rb")

        parse_scheme(self, stream)
