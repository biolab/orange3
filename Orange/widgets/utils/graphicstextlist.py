import bisect
import math
from typing import Optional, Union, Any, Iterable, List, Callable, cast

from AnyQt.QtCore import Qt, QSizeF, QEvent, QMarginsF, QPointF
from AnyQt.QtGui import QFont, QFontMetrics, QFontInfo, QPalette
from AnyQt.QtWidgets import (
    QGraphicsWidget, QSizePolicy, QGraphicsItemGroup, QGraphicsSimpleTextItem,
    QGraphicsItem, QGraphicsScene, QGraphicsSceneResizeEvent, QToolTip,
    QGraphicsSceneHelpEvent
)
from . import apply_all
from .graphicslayoutitem import scaled

__all__ = ["TextListWidget"]


class _FuncArray:
    __slots__ = ("func", "length")

    def __init__(self, func, length):
        self.func = func
        self.length = length

    def __getitem__(self, item):
        return self.func(item)

    def __len__(self):
        return self.length


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
            autoScale=False,
            elideMode=Qt.ElideNone,
            **kwargs: Any
    ) -> None:
        self.__items: List[str] = []
        self.__textitems: List[QGraphicsSimpleTextItem] = []
        self.__group: Optional[QGraphicsItemGroup] = None
        self.__spacing = 0
        self.__alignment = Qt.AlignmentFlag(alignment)
        self.__orientation = orientation
        self.__autoScale = autoScale
        # The effective font when autoScale is in effect
        self.__effectiveFont = QFont()
        self.__widthCache = {}
        self.__elideMode = elideMode
        sizePolicy = kwargs.pop(
            "sizePolicy", None)  # type: Optional[QSizePolicy]
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsWidget.ItemClipsChildrenToShape, True)
        sp = QSizePolicy(QSizePolicy.Preferred,
                         QSizePolicy.Preferred)
        sp.setWidthForHeight(True)
        self.setSizePolicy(sp)

        if sizePolicy is not None:
            self.setSizePolicy(sizePolicy)

        if parent is not None:
            self.setParentItem(parent)

        if items is not None:
            self.setItems(items)

    def setItems(self, items: Iterable[str]) -> None:
        """
        Set items for display

        Parameters
        ----------
        items: Iterable[str]
        """
        self.__clear()
        self.__items = list(items)
        self.__widthCache.clear()
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
            self.updateGeometry()

    def orientation(self) -> Qt.Orientation:
        return self.__orientation

    def clear(self) -> None:
        """
        Remove all items.
        """
        self.__clear()
        self.__items = []
        self.__widthCache.clear()
        self.updateGeometry()

    def count(self) -> int:
        """
        Return the number of items
        """
        return len(self.__items)

    def indexAt(self, pos: QPointF) -> Optional[int]:
        """
        Return the index of item at `pos`.
        """
        def brect(item):
            return item.mapRectToParent(item.boundingRect())

        if self.__orientation == Qt.Vertical:
            y = lambda pos: pos.y()
        else:
            y = lambda pos: pos.x()
        top = lambda idx: brect(items[idx]).top()
        bottom = lambda idx: brect(items[idx]).bottom()
        items = self.__textitems
        if not items:
            return None
        idx = bisect.bisect_right(_FuncArray(top, len(items)), y(pos)) - 1
        if idx == -1:
            idx = 0
        if top(idx) <= y(pos) <= bottom(idx):
            return idx
        else:
            return None

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

    def __width_for_font(self, font: QFont) -> float:
        """Return item width for the font"""
        key = font.key()
        if key in self.__widthCache:
            return self.__widthCache[key]
        fm = QFontMetrics(font)
        width = max((fm.boundingRect(text).width() for text in self.__items),
                    default=0)
        self.__widthCache[key] = width
        return width

    def __naturalsh(self) -> QSizeF:
        """Return the natural size hint (preferred sh with no constraints)."""
        fm = QFontMetrics(self.font())
        spacing = self.__spacing
        N = len(self.__items)
        width = self.__width_for_font(self.font())
        height = N * fm.height() + max(N - 1, 0) * spacing
        return QSizeF(width, height)

    def resizeEvent(self, event: QGraphicsSceneResizeEvent) -> None:
        super().resizeEvent(event)
        self.__layout()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.LayoutRequest:
            self.__layout()
        elif event.type() == QEvent.ContentsRectChange:
            self.__layout()
        elif event.type() == QEvent.GraphicsSceneHelp:
            self.helpEvent(cast(QGraphicsSceneHelpEvent, event))
            if event.isAccepted():
                return True
        return super().event(event)

    def helpEvent(self, event: QGraphicsSceneHelpEvent):
        idx = self.indexAt(self.mapFromScene(event.scenePos()))
        if idx is not None:
            rect = self.__textitems[idx].sceneBoundingRect()
            viewport = event.widget()
            view = viewport.parentWidget()
            rect = view.mapFromScene(rect).boundingRect()
            QToolTip.showText(event.screenPos(), self.__items[idx],
                              view, rect)
            event.setAccepted(True)

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.updateGeometry()
            if self.__autoScale:
                self.__layout()
            else:
                font = self.font()
                apply_all(self.__textitems, lambda it: it.setFont(font))

        elif event.type() == QEvent.PaletteChange:
            palette = self.palette()
            brush = palette.brush(QPalette.Text)
            for item in self.__textitems:
                item.setBrush(brush)
        super().changeEvent(event)

    def __setup(self) -> None:
        self.__clear()
        font = self.__effectiveFont if self.__autoScale else self.font()
        assert self.__group is None
        group = QGraphicsItemGroup()
        brush = self.palette().brush(QPalette.Text)
        for text in self.__items:
            t = QGraphicsSimpleTextItem(group)
            t.setBrush(brush)
            t.setFont(font)
            t.setText(text)
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
        font = self.font()
        fm = QFontMetrics(font)

        fontheight = fm.height()
        # the available vertical space
        vspace = crect.height() - (N - 1) * spacing
        cell_height = vspace / N

        if cell_height > fontheight:
            # use font height, adjust (widen) spacing.
            cell_height = fontheight
            spacing = (crect.height() - N * cell_height) / N
        elif self.__autoScale:
            # find a smaller font size to fit the height
            psize = effective_point_size_for_height(font, cell_height)
            font.setPointSizeF(psize)
            fm = QFontMetrics(font)
            fontheight = fm.height()

        if self.__autoScale and self.__effectiveFont != font:
            self.__effectiveFont = font
            apply_all(self.__textitems, lambda it: it.setFont(font))

        if self.__elideMode != Qt.ElideNone:
            if self.__orientation == Qt.Vertical:
                textwidth = math.ceil(crect.width())
            else:
                textwidth = math.ceil(crect.height())
            for text, item in zip(self.__items, self.__textitems):
                textelide = fm.elidedText(
                    text, self.__elideMode, textwidth, Qt.TextSingleLine
                )
                item.setText(textelide)

        advance = cell_height + spacing
        if align_vertical == Qt.AlignTop:
            align_dy = 0.
        elif align_vertical == Qt.AlignVCenter:
            align_dy = advance / 2.0 - fontheight / 2.0
        else:
            align_dy = advance - fontheight

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


def effective_point_size_for_height(
        font: QFont, height: float, step=0.25, minsize=1.
) -> float:
    font = QFont(font)
    start = max(math.ceil(height), minsize)
    font.setPointSizeF(start)
    fix = 0
    while QFontMetrics(font).height() > height and start - (fix + step) > minsize:
        fix += step
        font.setPointSizeF(start - fix)
    return QFontInfo(font).pointSizeF()


def effective_point_size_for_width(
        font: QFont, width: float, width_for_font: Callable[[QFont], float],
        step=1.0, minsize=1.,
) -> float:
    start = max(QFontInfo(font).pointSizeF(), minsize)
    font.setPointSizeF(start)
    fix = 0
    while width_for_font(font) > width and start - (fix + step) >= minsize:
        fix += step
        font.setPointSizeF(start - fix)
    return QFontInfo(font).pointSizeF()
