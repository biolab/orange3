"""
Node/Link layout.

"""
from operator import attrgetter, add

import numpy

import sip

from PyQt4.QtGui import QGraphicsObject
from PyQt4.QtCore import QRectF, QLineF, QTimer

from .items import NodeItem, LinkItem, SourceAnchorItem, SinkAnchorItem
from .items.utils import typed_signal_mapper, invert_permutation_indices, \
                         linspace
from functools import reduce

LinkItemSignalMapper = typed_signal_mapper(LinkItem)


def composition(f, g):
    """Return a composition of two functions
    """
    def fg(arg):
        return g(f(arg))
    return fg


class AnchorLayout(QGraphicsObject):
    def __init__(self, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsObject.ItemHasNoContents)

        self.__layoutPending = False
        self.__isActive = False
        self.__invalidatedAnchors = []
        self.__enabled = True

    def boundingRect(self):
        return QRectF()

    def activate(self):
        if self.isEnabled() and not self.__isActive:
            self.__isActive = True
            try:
                self._doLayout()
            finally:
                self.__isActive = False
                self.__layoutPending = False

    def isActivated(self):
        return self.__isActive

    def _doLayout(self):
        if not self.isEnabled():
            return

        scene = self.scene()
        items = scene.items()
        links = [item for item in items if isinstance(item, LinkItem)]
        point_pairs = [(link.sourceAnchor, link.sinkAnchor) for link in links]
        point_pairs.extend(map(reversed, point_pairs))
        to_other = dict(point_pairs)

        anchors = set(self.__invalidatedAnchors)

        for anchor_item in anchors:
            if sip.isdeleted(anchor_item):
                continue

            points = anchor_item.anchorPoints()
            anchor_pos = anchor_item.mapToScene(anchor_item.pos())
            others = [to_other[point] for point in points]

            if isinstance(anchor_item, SourceAnchorItem):
                others_angle = [-angle(anchor_pos, other.anchorScenePos())
                                for other in others]
            else:
                others_angle = [angle(other.anchorScenePos(), anchor_pos)
                                for other in others]

            indices = list(numpy.argsort(others_angle))
            # Invert the indices.
            indices = invert_permutation_indices(indices)

            positions = numpy.array(linspace(len(points)))
            positions = list(positions[indices])

            anchor_item.setAnchorPositions(positions)

        self.__invalidatedAnchors = []

    def invalidate(self):
        items = self.scene().items()
        nodes = [item for item in items is isinstance(item, NodeItem)]
        anchors = reduce(add,
                         [[node.outputAnchorItem, node.inputAnchorItem]
                          for node in nodes],
                         [])
        self.__invalidatedAnchors.extend(anchors)
        self.scheduleDelayedActivate()

    def invalidateLink(self, link):
        self.invalidateAnchorItem(link.sourceItem.outputAnchorItem)
        self.invalidateAnchorItem(link.sinkItem.inputAnchorItem)

        self.scheduleDelayedActivate()

    def invalidateNode(self, node):
        self.invalidateAnchorItem(node.inputAnchorItem)
        self.invalidateAnchorItem(node.outputAnchorItem)

        self.scheduleDelayedActivate()

    def invalidateAnchorItem(self, anchor):
        self.__invalidatedAnchors.append(anchor)

        scene = self.scene()
        if isinstance(anchor, SourceAnchorItem):
            links = scene.node_output_links(anchor.parentNodeItem())
            getter = composition(attrgetter("sinkItem"),
                                 attrgetter("inputAnchorItem"))
        elif isinstance(anchor, SinkAnchorItem):
            links = scene.node_input_links(anchor.parentNodeItem())
            getter = composition(attrgetter("sourceItem"),
                                 attrgetter("outputAnchorItem"))
        else:
            raise TypeError(type(anchor))

        self.__invalidatedAnchors.extend(map(getter, links))

        self.scheduleDelayedActivate()

    def scheduleDelayedActivate(self):
        if self.isEnabled() and not self.__layoutPending:
            self.__layoutPending = True
            QTimer.singleShot(0, self.__delayedActivate)

    def __delayedActivate(self):
        if self.__layoutPending:
            self.activate()


def angle(point1, point2):
    """Return the angle between the two points in range from -180 to 180.
    """
    angle = QLineF(point1, point2).angle()
    if angle > 180:
        return angle - 360
    else:
        return angle
