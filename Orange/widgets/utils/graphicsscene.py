from AnyQt.QtCore import Qt
from AnyQt.QtGui import QTransform
from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsSceneHelpEvent,QGraphicsView, QToolTip
)

__all__ = [
    "GraphicsScene"
]


class GraphicsScene(QGraphicsScene):
    """
    A QGraphicsScene with better tool tip event dispatch.
    """
    def helpEvent(self, event: QGraphicsSceneHelpEvent) -> None:
        """
        Reimplemented.

        Send the help event to every graphics item that is under the event's
        scene position (default `QGraphicsScene` only dispatches help events
        to `QGraphicsProxyWidget`s.
        """
        widget = event.widget()
        if widget is not None and isinstance(widget.parentWidget(),
                                             QGraphicsView):
            view = widget.parentWidget()
            deviceTransform = view.viewportTransform()
        else:
            deviceTransform = QTransform()
        items = self.items(
            event.scenePos(), Qt.IntersectsItemShape, Qt.DescendingOrder,
            deviceTransform,
        )
        text = None
        event.setAccepted(False)
        for item in items:
            self.sendEvent(item, event)
            if event.isAccepted():
                return
            elif item.toolTip():
                text = item.toolTip()
                break

        if text is not None:
            QToolTip.showText(event.screenPos(), text, event.widget())
