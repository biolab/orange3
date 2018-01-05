"""
Signal Manager
==============

A SignalManager instance handles the runtime signal propagation between
widgets in a scheme.


"""

import logging
import warnings
from collections import namedtuple, defaultdict, deque
from operator import attrgetter
from functools import partial, reduce

from AnyQt.QtCore import QObject, QCoreApplication, QEvent, QTimer
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from .scheme import SchemeNode, SchemeLink

log = logging.getLogger(__name__)


_Signal = namedtuple(
    "_Signal",
    ["link",     # link on which the signal is sent
     "value",    # signal value
     "id"])      # signal id


is_enabled = attrgetter("enabled")

MAX_CONCURRENT = 1


class SignalManager(QObject):
    """
    Handle all runtime signal propagation for a :clas:`Scheme` instance.
    The scheme must be passed to the constructor and will become the parent
    of this object. Furthermore this should happen before any items
    (nodes, links) are added to the scheme.

    """
    Running, Stoped, Paused, Error = range(4)
    """SignalManger state flags."""

    Waiting, Processing = range(2)
    """SignalManager runtime state flags."""

    stateChanged = Signal(int)
    """Emitted when the state of the signal manager changes."""

    updatesPending = Signal()
    """Emitted when signals are added to the queue."""

    processingStarted = Signal([], [SchemeNode])
    """Emitted right before a `SchemeNode` instance has its inputs
    updated.
    """

    processingFinished = Signal([], [SchemeNode])
    """Emitted right after a `SchemeNode` instance has had its inputs
    updated.
    """

    runtimeStateChanged = Signal(int)
    """Emitted when `SignalManager`'s runtime state changes."""

    def __init__(self, scheme):
        assert(scheme)
        QObject.__init__(self, scheme)
        self._input_queue = []

        # mapping a node to it's current outputs
        # {node: {channel: {id: signal_value}}}
        self._node_outputs = {}

        self.__state = SignalManager.Running
        self.__runtime_state = SignalManager.Waiting

        # A flag indicating if UpdateRequest event should be rescheduled
        self.__reschedule = False
        self.__update_timer = QTimer(self, interval=100, singleShot=True)
        self.__update_timer.timeout.connect(self.__process_next)

    def _can_process(self):
        """
        Return a bool indicating if the manger can enter the main
        processing loop.

        """
        return self.__state not in [SignalManager.Error, SignalManager.Stoped]

    def scheme(self):
        """
        Return the parent class:`Scheme` instance.
        """
        return self.parent()

    def start(self):
        """
        Start the update loop.

        .. note:: The updates will not happen until the control reaches
                  the Qt event loop.

        """
        if self.__state != SignalManager.Running:
            self.__state = SignalManager.Running
            self.stateChanged.emit(SignalManager.Running)
            self._update()

    def stop(self):
        """
        Stop the update loop.

        .. note:: If the `SignalManager` is currently in `process_queues` it
                  will still update all current pending signals, but will not
                  re-enter until `start()` is called again

        """
        if self.__state != SignalManager.Stoped:
            self.__state = SignalManager.Stoped
            self.stateChanged.emit(SignalManager.Stoped)
            self.__update_timer.stop()

    def pause(self):
        """
        Pause the updates.

        """
        if self.__state != SignalManager.Paused:
            self.__state = SignalManager.Paused
            self.stateChanged.emit(SignalManager.Paused)
            self.__update_timer.stop()

    def resume(self):
        if self.__state == SignalManager.Paused:
            self.__state = SignalManager.Running
            self.stateChanged.emit(self.__state)
            self._update()

    def step(self):
        if self.__state == SignalManager.Paused:
            self.process_queued()

    def state(self):
        """
        Return the current state.
        """
        return self.__state

    def _set_runtime_state(self, state):
        """
        Set the runtime state.

        Should only be called by `SignalManager` implementations.

        """
        if self.__runtime_state != state:
            self.__runtime_state = state
            self.runtimeStateChanged.emit(self.__runtime_state)

    def runtime_state(self):
        """
        Return the runtime state. This can be `SignalManager.Waiting`
        or `SignalManager.Processing`.

        """
        return self.__runtime_state

    def on_node_removed(self, node):
        # remove all pending input signals for node so we don't get
        # stale references in process_node.
        # NOTE: This does not remove output signals for this node. In
        # particular the final 'None' will be delivered to the sink
        # nodes even after the source node is no longer in the scheme.
        log.info("Node %r removed. Removing pending signals.",
                 node.title)
        self.remove_pending_signals(node)

        del self._node_outputs[node]

    def on_node_added(self, node):
        self._node_outputs[node] = defaultdict(dict)

    def link_added(self, link):
        # push all current source values to the sink
        link.set_runtime_state(SchemeLink.Empty)
        if link.enabled:
            log.info("Link added (%s). Scheduling signal data update.", link)
            self._schedule(self.signals_on_link(link))
            self._update()

        link.enabled_changed.connect(self.link_enabled_changed)

    def link_removed(self, link):
        # purge all values in sink's queue
        log.info("Link removed (%s). Scheduling signal data purge.", link)
        self.purge_link(link)
        link.enabled_changed.disconnect(self.link_enabled_changed)

    def link_enabled_changed(self, enabled):
        if enabled:
            link = self.sender()
            log.info("Link %s enabled. Scheduling signal data update.", link)
            self._schedule(self.signals_on_link(link))

    def signals_on_link(self, link):
        """
        Return _Signal instances representing the current values
        present on the link.

        """
        items = self.link_contents(link)
        signals = []

        for key, value in list(items.items()):
            signals.append(_Signal(link, value, key))

        return signals

    def link_contents(self, link):
        """
        Return the contents on link.
        """
        node, channel = link.source_node, link.source_channel

        if node in self._node_outputs:
            return self._node_outputs[node][channel]
        else:
            # if the the node was already removed it's tracked outputs in
            # _node_outputs are cleared, however the final 'None' signal
            # deliveries for the link are left in the _input_queue.
            pending = [sig for sig in self._input_queue
                       if sig.link is link]
            return {sig.id: sig.value for sig in pending}

    def send(self, node, channel, value, id):
        """
        """
        log.debug("%r sending %r (id: %r) on channel %r",
                  node.title, type(value), id, channel.name)

        scheme = self.scheme()

        self._node_outputs[node][channel][id] = value

        links = scheme.find_links(source_node=node, source_channel=channel)
        links = list(filter(is_enabled, links))

        signals = []
        for link in links:
            signals.append(_Signal(link, value, id))

        self._schedule(signals)

    def purge_link(self, link):
        """
        Purge the link (send None for all ids currently present)
        """
        contents = self.link_contents(link)
        ids = list(contents.keys())
        signals = [_Signal(link, None, id) for id in ids]

        self._schedule(signals)

    def _schedule(self, signals):
        """
        Schedule a list of :class:`_Signal` for delivery.
        """
        self._input_queue.extend(signals)

        for link in {sig.link for sig in signals}:
            # update the SchemeLink's runtime state flags
            contents = self.link_contents(link)
            if any(value is not None for value in contents.values()):
                state = SchemeLink.Active
            else:
                state = SchemeLink.Empty
            link.set_runtime_state(state | SchemeLink.Pending)

        if signals:
            self.updatesPending.emit()

        self._update()

    def _update_link(self, link):
        """
        Schedule update of a single link.
        """
        signals = self.signals_on_link(link)
        self._schedule(signals)

    def process_queued(self, max_nodes=None):
        """
        Process queued signals.

        Take one node node from the pending input queue and deliver
        all scheduled signals.
        """
        if not (max_nodes is None or max_nodes == 1):
            warnings.warn(
                "`max_nodes` is deprecated and unused (will always equal 1)",
                DeprecationWarning, stacklevel=2)

        if self.__runtime_state == SignalManager.Processing:
            raise RuntimeError("Cannot re-enter 'process_queued'")

        if not self._can_process():
            raise RuntimeError("Can't process in state %i" % self.__state)

        log.info("SignalManager: Processing queued signals")

        node_update_front = self.node_update_front()
        log.debug("SignalManager: Nodes eligible for update %s",
                  [node.title for node in node_update_front])

        if node_update_front:
            node = node_update_front[0]
            self._set_runtime_state(SignalManager.Processing)
            try:
                self.process_node(node)
            finally:
                self._set_runtime_state(SignalManager.Waiting)

    def process_node(self, node):
        """
        Process pending input signals for `node`.
        """
        signals_in = self.pending_input_signals(node)
        self.remove_pending_signals(node)

        signals_in = self.compress_signals(signals_in)

        log.debug("Processing %r, sending %i signals.",
                  node.title, len(signals_in))
        # Clear the link's pending flag.
        for link in {sig.link for sig in signals_in}:
            link.set_runtime_state(link.runtime_state() & ~SchemeLink.Pending)

        assert ({sig.link for sig in self._input_queue}
                .intersection({sig.link for sig in signals_in}) == set([]))
        self.processingStarted.emit()
        self.processingStarted[SchemeNode].emit(node)
        try:
            self.send_to_node(node, signals_in)
        finally:
            self.processingFinished.emit()
            self.processingFinished[SchemeNode].emit(node)

    def compress_signals(self, signals):
        """
        Compress a list of :class:`_Signal` instances to be delivered.

        The base implementation returns the list unmodified.

        """
        return signals

    def send_to_node(self, node, signals):
        """
        Abstract. Reimplement in subclass.

        Send/notify the :class:`SchemeNode` instance (or whatever
        object/instance it is a representation of) that it has new inputs
        as represented by the signals list (list of :class:`_Signal`).

        """
        raise NotImplementedError

    def is_pending(self, node):
        """
        Is `node` (class:`SchemeNode`) scheduled for processing (i.e.
        it has incoming pending signals).

        """
        return node in [signal.link.sink_node for signal in self._input_queue]

    def pending_nodes(self):
        """
        Return a list of pending nodes.

        The nodes are returned in the order they were enqueued for
        signal delivery.

        Returns
        -------
        nodes : List[SchemeNode]
        """
        return list(unique(sig.link.sink_node for sig in self._input_queue))

    def pending_input_signals(self, node):
        """
        Return a list of pending input signals for node.
        """
        return [signal for signal in self._input_queue
                if node is signal.link.sink_node]

    def remove_pending_signals(self, node):
        """
        Remove pending signals for `node`.
        """
        for signal in self.pending_input_signals(node):
            try:
                self._input_queue.remove(signal)
            except ValueError:
                pass

    def blocking_nodes(self):
        """
        Return a list of nodes in a blocking state.
        """
        scheme = self.scheme()
        return [node for node in scheme.nodes if self.is_blocking(node)]

    def is_blocking(self, node):
        return False

    def node_update_front(self):
        """
        Return a list of nodes on the update front, i.e. nodes scheduled for
        an update that have no ancestor which is either itself scheduled
        for update or is in a blocking state)

        .. note::
            The node's ancestors are only computed over enabled links.

        """
        scheme = self.scheme()

        blocking_nodes = set(self.blocking_nodes())

        dependents = partial(dependent_nodes, scheme)

        blocked_nodes = reduce(set.union,
                               map(dependents, blocking_nodes),
                               set(blocking_nodes))

        pending = self.pending_nodes()
        pending_downstream = reduce(set.union,
                                    map(dependents, pending),
                                    set())

        log.debug("Pending nodes: %s", pending)
        log.debug("Blocking nodes: %s", blocking_nodes)

        noneligible = pending_downstream | blocked_nodes
        return [node for node in pending if node not in noneligible]

    @Slot()
    def __process_next(self):
        if not self.__state == SignalManager.Running:
            log.debug("Received 'UpdateRequest' while not in 'Running' state")
            return

        if self.__runtime_state == SignalManager.Processing:
            # This happens if someone calls QCoreApplication.processEvents
            # from the signal handlers.
            # A `__process_next` must be rescheduled when exiting
            # process_queued.
            log.warning("Received 'UpdateRequest' while in 'process_queued'. "
                        "An update will be re-scheduled when exiting the "
                        "current update.")
            self.__reschedule = True
            return

        nbusy = len(self.blocking_nodes())
        log.info("'UpdateRequest' event, queued signals: %i, nbusy: %i "
                 "(MAX_CONCURRENT: %i)",
                 len(self._input_queue), nbusy, MAX_CONCURRENT)

        if self._input_queue and nbusy < MAX_CONCURRENT:
            self.process_queued()

        if self.__reschedule and self.__state == SignalManager.Running:
            self.__reschedule = False
            log.debug("Rescheduling signal update")
            self.__update_timer.start()

        nbusy = len(self.blocking_nodes())
        if self.node_update_front() and nbusy < MAX_CONCURRENT:
            log.debug("More nodes are eligible for an update. "
                      "Scheduling another update.")
            self._update()

    def _update(self):
        """
        Schedule processing at a later time.
        """
        if self.__state == SignalManager.Running and \
                not self.__update_timer.isActive():
            self.__update_timer.start()


def can_enable_dynamic(link, value):
    """
    Can the a dynamic `link` (:class:`SchemeLink`) be enabled for`value`.
    """
    return isinstance(value, link.sink_type())


def compress_signals(signals):
    """
    Compress a list of signals.
    """
    groups = group_by_all(reversed(signals),
                          key=lambda sig: (sig.link, sig.id))
    signals = []

    def has_none(signals):
        return any(sig.value is None for sig in signals)

    for (link, id), signals_grouped in groups:
        if len(signals_grouped) > 1 and has_none(signals_grouped[1:]):
            signals.append(signals_grouped[0])
            signals.append(_Signal(link, None, id))
        else:
            signals.append(signals_grouped[0])

    return list(reversed(signals))


def dependent_nodes(scheme, node):
    """
    Return a list of all nodes (in breadth first order) in `scheme` that
    are dependent on `node`,

    .. note::
        This does not include nodes only reachable by disables links.

    """
    def expand(node):
        return [link.sink_node
                for link in scheme.find_links(source_node=node)
                if link.enabled]

    nodes = list(traverse_bf(node, expand))
    assert nodes[0] is node
    # Remove the first item (`node`).
    return nodes[1:]


def traverse_bf(start, expand):
    queue = deque([start])
    visited = set()
    while queue:
        item = queue.popleft()
        if item not in visited:
            yield item
            visited.add(item)
            queue.extend(expand(item))


def group_by_all(sequence, key=None):
    order_seen = []
    groups = {}
    for item in sequence:
        if key is not None:
            item_key = key(item)
        else:
            item_key = item
        if item_key in groups:
            groups[item_key].append(item)
        else:
            groups[item_key] = [item]
            order_seen.append(item_key)

    return [(key, groups[key]) for key in order_seen]


def unique(iterable):
    """
    Return unique elements of `iterable` while preserving their order.

    Parameters
    ----------
    iterable : Iterable[Hashable]


    Returns
    -------
    unique : Iterable
        Unique elements from `iterable`.
    """
    seen = set()

    def observed(el):
        observed = el in seen
        seen.add(el)
        return observed

    return (el for el in iterable if not observed(el))
