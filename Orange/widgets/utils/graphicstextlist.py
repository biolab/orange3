from typing import Optional, Union, Any, Iterable, List

from AnyQt.QtCore import Qt, QSizeF, QEvent, QMarginsF
from AnyQt.QtGui import QFontMetrics
from AnyQt.QtWidgets import (
    QGraphicsWidget, QSizePolicy, QGraphicsItemGroup, QGraphicsSimpleTextItem,
    QGraphicsItem, QGraphicsScene, QWIDGETSIZE_MAX,
)

__all__ = ["TextListWidget"]


class TextListWidget(QGraphicsWidget):
    """
    A linear text list widget.

    Displays a list of uniformly spaced text lines.

    Parameters
    ----------
    parent: Optional[QGraphicsItem]
    items: Iterable[str]
    alignment: Qt.Alignment
    orientation: Qt.Orientation
    """
    def __init__(
            self,
            parent: Optional[QGraphicsItem] = None,
            items: Iterable[str] = (),
            alignment: Union[Qt.AlignmentFlag, Qt.Alignment] = Qt.AlignLeading,
            orientation: Qt.Orientation = Qt.Vertical,
            **kwargs: Any
    ) -> None:
        sizePolicy = kwargs.pop(
            "sizePolicy", None)  # type: Optional[QSizePolicy]
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsWidget.ItemClipsChildrenToShape, True)
        self.__items: List[str] = []
        self.__textitems: List[QGraphicsSimpleTextItem] = []
        self.__group: Optional[QGraphicsItemGroup] = None
        self.__spacing = 0
        self.__alignment = Qt.AlignmentFlag(alignment)
        self.__orientation = orientation

        sp = QSizePolicy(QSizePolicy.Preferred,
                         QSizePolicy.Preferred)
        sp.setWidthForHeight(True)
        self.setSizePolicy(sp)

        if items is not None:
            self.setItems(items)

        if sizePolicy is not None:
            self.setSizePolicy(sizePolicy)

        if parent is not None:
            self.setParentItem(parent)

    def setItems(self, items: Iterable[str]) -> None:
        """
        Set items for display

        Parameters
        ----------
        items: Iterable[str]
        """
        self.__clear()
        self.__items = list(items)
        self.__setup()
        self.__layout()
        self.updateGeometry()

    def setAlignment(self, alignment: Qt.AlignmentFlag) -> None:
        """
        Set the text item's alignment.
        """
        if self.__alignment != alignment:
            self.__alignment = alignment
            self.__layout()

    def alignment(self) -> Qt.AlignmentFlag:
        """Return the text item's alignment."""
        return self.__alignment

    def setOrientation(self, orientation: Qt.Orientation) -> None:
        """
        Set text orientation.

        If Qt.Vertical items are put in a vertical layout
        if Qt.Horizontal the n items are drawn rotated 90 degrees and laid out
        horizontally with first text items's top corner in the bottom left
        of `self.geometry()`.

        Parameters
        ----------
        orientation: Qt.Orientation
        """
        if self.__orientation != orientation:
            self.__orientation = orientation
            self.__layout()

    def orientation(self) -> Qt.Orientation:
        return self.__orientation

    def clear(self) -> None:
        """
        Remove all items.
        """
        self.__clear()
        self.__items = []
        self.updateGeometry()

    def count(self) -> int:
        """
        Return the number of items
        """
        return len(self.__items)

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF()) -> QSizeF:
        """Reimplemented."""
        if which == Qt.PreferredSize:
            sh = self.__naturalsh()
            if self.__orientation == Qt.Vertical:
                if 0 < constraint.height() < sh.height():
                    sh = scaled(sh, constraint, Qt.KeepAspectRatioByExpanding)
            else:
                sh = sh.transposed()
                if 0 < constraint.width() < sh.width():
                    sh = scaled(sh, constraint, Qt.KeepAspectRatioByExpanding)
        else:
            sh = super().sizeHint(which, constraint)
        return sh

    def __naturalsh(self) -> QSizeF:
        """Return the natural size hint (preferred sh with no constraints)."""
        fm = QFontMetrics(self.font())
        spacing = self.__spacing
        N = len(self.__items)
        width = max((fm.width(text) for text in self.__items),
                    default=0)
        height = N * fm.height() + max(N - 1, 0) * spacing
        return QSizeF(width, height)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.LayoutRequest:
            self.__layout()
        elif event.type() == QEvent.GraphicsSceneResize:
            self.__layout()
        elif event.type() == QEvent.ContentsRectChange:
            self.__layout()
        return super().event(event)

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.updateGeometry()
            font = self.font()
            for item in self.__textitems:
                item.setFont(font)
        super().changeEvent(event)

    def __setup(self) -> None:
        self.__clear()
        font = self.font()
        assert self.__group is None
        group = QGraphicsItemGroup()
        for text in self.__items:
            t = QGraphicsSimpleTextItem(group)
            t.setFont(font)
            t.setText(text)
            t.setToolTip(text)
            t.setData(0, text)
            self.__textitems.append(t)
        group.setParentItem(self)
        self.__group = group

    def __layout(self) -> None:
        margins = QMarginsF(*self.getContentsMargins())
        if self.__orientation == Qt.Horizontal:
            # transposed margins
            margins = QMarginsF(
                margins.bottom(), margins.left(), margins.top(), margins.right()
            )
            crect = self.rect().transposed().marginsRemoved(margins)
        else:
            crect = self.rect().marginsRemoved(margins)

        spacing = self.__spacing

        align_horizontal = self.__alignment & Qt.AlignHorizontal_Mask
        align_vertical = self.__alignment & Qt.AlignVertical_Mask
        if align_vertical == 0:
            align_vertical = Qt.AlignTop
        if align_horizontal == 0:
            align_horizontal = Qt.AlignLeft

        N = len(self.__items)

        if not N:
            return

        assert self.__group is not None

        fm = QFontMetrics(self.font())
        naturalheight = fm.height()
        cell_height = (crect.height() - (N - 1) * spacing) / N

        if cell_height > naturalheight and N > 1:
            cell_height = naturalheight
            spacing = (crect.height() - N * cell_height) / N

        advance = cell_height + spacing
        if align_vertical == Qt.AlignTop:
            align_dy = 0.
        elif align_vertical == Qt.AlignVCenter:
            align_dy = advance / 2.0 - naturalheight / 2.0
        else:
            align_dy = advance - naturalheight

        if align_horizontal == Qt.AlignLeft:
            for i, item in enumerate(self.__textitems):
                item.setPos(crect.left(), crect.top() + i * advance + align_dy)
        elif align_horizontal == Qt.AlignHCenter:
            for i, item in enumerate(self.__textitems):
                item.setPos(
                    crect.center().x() - item.boundingRect().width() / 2,
                    crect.top() + i * advance + align_dy
                )
        else:
            for i, item in enumerate(self.__textitems):
                item.setPos(
                    crect.right() - item.boundingRect().width(),
                    crect.top() + i * advance + align_dy
                )

        if self.__orientation == Qt.Vertical:
            self.__group.setRotation(0)
            self.__group.setPos(0, 0)
        else:
            self.__group.setRotation(-90)
            self.__group.setPos(self.rect().bottomLeft())

    def __clear(self) -> None:
        def remove(items: Iterable[QGraphicsItem],
                   scene: Optional[QGraphicsScene]):
            for item in items:
                if scene is not None:
                    scene.removeItem(item)
                else:
                    item.setParentItem(None)
        self.__textitems = []
        if self.__group is not None:
            remove([self.__group], self.scene())
            self.__group = None


def scaled(size: QSizeF, constraint: QSizeF, mode=Qt.KeepAspectRatio) -> QSizeF:
    """
    Return size scaled to fit in the constrains using the aspect mode `mode`.

    If  width or height of constraint are negative they are ignored,
    ie. the result is not constrained in that dimension.
    """
    size, constraint = QSizeF(size), QSizeF(constraint)
    if constraint.width() < 0 and constraint.height() < 0:
        return size
    if mode == Qt.IgnoreAspectRatio:
        if constraint.width() >= 0:
            size.setWidth(constraint.width())
        if constraint.height() >= 0:
            size.setHeight(constraint.height())
    elif mode == Qt.KeepAspectRatio:
        if constraint.width() < 0:
            constraint.setWidth(QWIDGETSIZE_MAX)
        if constraint.height() < 0:
            constraint.setHeight(QWIDGETSIZE_MAX)
        size.scale(constraint, mode)
    elif mode == Qt.KeepAspectRatioByExpanding:
        if constraint.width() < 0:
            constraint.setWidth(0)
        if constraint.height() < 0:
            constraint.setHeight(0)
        size.scale(constraint, mode)
    return size
