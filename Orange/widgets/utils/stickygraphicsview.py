import sys
import math

from AnyQt.QtCore import (
    Qt, QRectF, QEvent, QCoreApplication, QObject, QPointF, QRect
)
from AnyQt.QtGui import QBrush, QPalette, QTransform, QPolygonF
from AnyQt.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, QSizePolicy,
    QScrollBar, QGraphicsDropShadowEffect
)

from orangewidget.utils.overlay import OverlayWidget

__all__ = [
    "StickyGraphicsView"
]


class _OverlayWidget(OverlayWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        palette = self.palette()
        ds = QGraphicsDropShadowEffect(
            parent=self,
            objectName="sticky-view-shadow",
            color=palette.color(QPalette.Foreground),
            blurRadius=15,
            offset=QPointF(0, 0),
            enabled=True
        )
        self.setGraphicsEffect(ds)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:
            effect = self.findChild(
                QGraphicsDropShadowEffect, "sticky-view-shadow")
            if effect is not None:
                palette = self.palette()
                effect.setColor(palette.color(QPalette.Foreground))

    def eventFilter(self, recv: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Show, QEvent.Hide) and recv is self.widget():
            return False
        else:
            return super().eventFilter(recv, event)


class _HeaderGraphicsView(QGraphicsView):
    def viewportEvent(self, event: QEvent) -> bool:
        if event.type() == QEvent.Wheel:
            # delegate wheel events to parent StickyGraphicsView
            parent = self.parent().parent().parent()
            if isinstance(parent, StickyGraphicsView):
                QCoreApplication.sendEvent(parent.viewport(), event)
                if event.isAccepted():
                    return True
        return super().viewportEvent(event)


class StickyGraphicsView(QGraphicsView):
    """
    A graphics view with sticky header/footer views.

    Set the scene rect of the header/footer geometry with
    setHeaderRect/setFooterRect. When scrolling they will be displayed
    top/bottom of the viewport.
    """
    def __init__(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], QGraphicsScene):
            scene, args = args[0], args[1:]
        else:
            scene = None
        super().__init__(*args, **kwargs)
        self.__headerRect = QRectF()
        self.__footerRect = QRectF()
        self.__headerView: QGraphicsView = ...
        self.__footerView: QGraphicsView = ...
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setupViewport(self.viewport())

        if scene is not None:
            self.setScene(scene)

    def setHeaderSceneRect(self, rect: QRectF) -> None:
        """
        Set the header scene rect.

        Parameters
        ----------
        rect : QRectF
        """
        if self.__headerRect != rect:
            self.__headerRect = QRectF(rect)
            self.__updateHeader()

    def headerSceneRect(self) -> QRectF:
        return QRectF(self.__headerRect)

    def setFooterSceneRect(self, rect: QRectF) -> None:
        """
        Set the footer scene rect.

        Parameters
        ----------
        rect : QRectF
        """
        if self.__footerRect != rect:
            self.__footerRect = QRectF(rect)
            self.__updateFooter()

    def footerSceneRect(self) -> QRectF:
        return QRectF(self.__footerRect)

    def setScene(self, scene: QGraphicsScene) -> None:
        """Reimplemented"""
        super().setScene(scene)
        self.headerView().setScene(scene)
        self.footerView().setScene(scene)
        self.__headerRect = QRectF()
        self.__footerRect = QRectF()

    def setupViewport(self, widget: QWidget) -> None:
        """Reimplemented"""
        super().setupViewport(widget)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header = _HeaderGraphicsView(
            objectName="sticky-header-view", sizePolicy=sp,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            alignment=self.alignment()
        )
        header.setFocusProxy(self)
        header.viewport().installEventFilter(self)
        header.setFrameStyle(QGraphicsView.NoFrame)

        footer = _HeaderGraphicsView(
            objectName="sticky-footer-view", sizePolicy=sp,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            alignment=self.alignment()
        )
        footer.setFocusProxy(self)
        footer.viewport().installEventFilter(self)
        footer.setFrameStyle(QGraphicsView.NoFrame)

        over = _OverlayWidget(
            widget, objectName="sticky-header-overlay-container",
            alignment=Qt.AlignTop,
            sizePolicy=sp,
            visible=False,
        )
        over.setLayout(QVBoxLayout(margin=0))
        over.layout().addWidget(header)
        over.setWidget(widget)

        over = _OverlayWidget(
            widget, objectName="sticky-footer-overlay-container",
            alignment=Qt.AlignBottom,
            sizePolicy=sp,
            visible=False
        )
        over.setLayout(QVBoxLayout(margin=0))
        over.layout().addWidget(footer)
        over.setWidget(widget)

        def bind(source: QScrollBar, target: QScrollBar) -> None:
            # bind target scroll bar to `source` (range and value).
            target.setRange(source.minimum(), source.maximum())
            target.setValue(source.value())
            source.rangeChanged.connect(target.setRange)
            source.valueChanged.connect(target.setValue)

        hbar = self.horizontalScrollBar()
        footer_hbar = footer.horizontalScrollBar()
        header_hbar = header.horizontalScrollBar()

        bind(hbar, footer_hbar)
        bind(hbar, header_hbar)

        self.__headerView = header
        self.__footerView = footer
        self.__updateView(header, self.__footerRect)
        self.__updateView(footer, self.__footerRect)

    def headerView(self) -> QGraphicsView:
        """
        Return the header view.

        Returns
        -------
        view: QGraphicsView
        """
        return self.__headerView

    def footerView(self) -> QGraphicsView:
        """
        Return the footer view.

        Returns
        -------
        view: QGraphicsView
        """
        return self.__footerView

    def setTransform(self, matrix: QTransform, combine: bool = False) -> None:
        self.__headerView.setTransform(matrix, combine)
        self.__footerView.setTransform(matrix, combine)
        super().setTransform(matrix, combine)
        self.__updateFooter()
        self.__updateHeader()

    def __updateView(self, view: QGraphicsView, rect: QRectF) -> None:
        view.setSceneRect(rect)
        viewrect = view.mapFromScene(rect).boundingRect()
        view.setFixedHeight(int(math.ceil(viewrect.height())))
        container = view.parent()
        if rect.isEmpty():
            container.setVisible(False)
            return
        # map the rect to (main) viewport coordinates
        viewrect = qgraphicsview_map_rect_from_scene(self, rect).boundingRect()
        viewrect = qrectf_to_inscribed_rect(viewrect)
        viewportrect = self.viewport().rect()
        visible = (viewrect.top() < viewportrect.top() or
                   viewrect.y() + viewrect.height() > viewportrect.y() + viewportrect.height())
        container.setVisible(visible)
        # force immediate layout of the container overlay
        QCoreApplication.sendEvent(container, QEvent(QEvent.LayoutRequest))

    def __updateHeader(self) -> None:
        view = self.headerView()
        self.__updateView(view, self.__headerRect)

    def __updateFooter(self) -> None:
        view = self.footerView()
        self.__updateView(view, self.__footerRect)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        """Reimplemented."""
        super().scrollContentsBy(dx, dy)
        self.__updateFooter()
        self.__updateHeader()

    def viewportEvent(self, event: QEvent) -> bool:
        """Reimplemented."""
        if event.type() == QEvent.Resize:
            self.__updateHeader()
            self.__updateFooter()
        return super().viewportEvent(event)


def qgraphicsview_map_rect_from_scene(
        view: QGraphicsView, rect: QRectF
) -> QPolygonF:
    """Like QGraphicsView.mapFromScene(QRectF) but returning a QPolygonF
    (without rounding).
    """
    tr = view.viewportTransform()
    p1 = tr.map(rect.topLeft())
    p2 = tr.map(rect.topRight())
    p3 = tr.map(rect.bottomRight())
    p4 = tr.map(rect.bottomLeft())
    return QPolygonF([p1, p2, p3, p4])


def qrectf_to_inscribed_rect(rect: QRectF) -> QRect:
    """
    Return the largest integer QRect such that it is completely contained in
    `rect`.
    """
    xmin = int(math.ceil(rect.x()))
    xmax = int(math.floor(rect.right()))
    ymin = int(math.ceil(rect.top()))
    ymax = int(math.floor(rect.bottom()))
    return QRect(xmin, ymin, max(xmax - xmin, 0), max(ymax - ymin, 0))


def main(args):  # pragma: no cover
    # pylint: disable=import-outside-toplevel,protected-access
    from AnyQt.QtWidgets import QApplication, QAction
    from AnyQt.QtGui import QKeySequence
    app = QApplication(args)
    view = StickyGraphicsView()
    scene = QGraphicsScene(view)
    scene.setBackgroundBrush(QBrush(Qt.lightGray, Qt.CrossPattern))
    view.setScene(scene)
    scene.addRect(
        QRectF(0, 0, 300, 20), Qt.red, QBrush(Qt.red, Qt.BDiagPattern))
    scene.addRect(QRectF(0, 25, 300, 100))
    scene.addRect(
        QRectF(0, 130, 300, 20),
        Qt.darkGray, QBrush(Qt.darkGray, Qt.BDiagPattern)
    )
    view.setHeaderSceneRect(QRectF(0, 0, 300, 20))
    view.setFooterSceneRect(QRectF(0, 130, 300, 20))
    view.show()
    zoomin = QAction("Zoom in", view, shortcut=QKeySequence.ZoomIn)
    zoomout = QAction("Zoom out", view, shortcut=QKeySequence.ZoomOut)
    zoomreset = QAction(
        "Reset", view, shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0)
    )
    view._zoom = 100

    def set_zoom(zoom):
        if view._zoom != zoom:
            view._zoom = zoom
            view.setTransform(
                QTransform.fromScale(*(view._zoom / 100,) * 2)
            )
            zoomout.setEnabled(zoom >= 20)
            zoomin.setEnabled(zoom <= 300)

    @zoomin.triggered.connect
    def _():
        set_zoom(view._zoom + 10)

    @zoomout.triggered.connect
    def _():
        set_zoom(view._zoom - 10)

    @zoomreset.triggered.connect
    def _():
        set_zoom(100)

    view.addActions([
        zoomin, zoomout, zoomreset
    ])

    return app.exec()


if __name__ == "__main__":
    main(sys.argv)
