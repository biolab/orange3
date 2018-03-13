"""
Preview Browser Widget.

"""
import os
from xml.sax.saxutils import escape

from AnyQt.QtWidgets import (
    QWidget, QLabel, QAction, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QStyleOption, QStylePainter
)
from AnyQt.QtSvg import QSvgWidget
from AnyQt.QtCore import (
    Qt, QSize, QByteArray, QModelIndex, QEvent
)
from AnyQt.QtCore import pyqtSignal as Signal

from ..utils import check_type
from ..gui.dropshadow import DropShadowFrame
from ..gui.iconview import LinearIconView
from . import previewmodel


NO_PREVIEW_SVG = """

"""


# Default description template
DESCRIPTION_TEMPLATE = """
<h3 class=item-heading>{name}</h3>
<p class=item-description>
{description}
</p>

"""

PREVIEW_SIZE = (440, 295)


class TextLabel(QWidget):
    """A plain text label widget with support for elided text.
    """
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.setSizePolicy(QSizePolicy.Expanding,
                           QSizePolicy.Preferred)

        self.__text = ""
        self.__textElideMode = Qt.ElideMiddle
        self.__sizeHint = None
        self.__alignment = Qt.AlignLeft | Qt.AlignVCenter

    def setText(self, text):
        """Set the `text` string to display.
        """
        check_type(text, str)
        if self.__text != text:
            self.__text = str(text)
            self.__update()

    def text(self):
        """Return the text
        """
        return self.__text

    def setTextElideMode(self, mode):
        """Set elide mode (`Qt.TextElideMode`)
        """
        if self.__textElideMode != mode:
            self.__textElideMode = mode
            self.__update()

    def elideMode(self):
        return self.__elideMode

    def setAlignment(self, align):
        """Set text alignment (`Qt.Alignment`).
        """
        if self.__alignment != align:
            self.__alignment = align
            self.__update()

    def sizeHint(self):
        if self.__sizeHint is None:
            option = QStyleOption()
            option.initFrom(self)
            metrics = option.fontMetrics

            self.__sizeHint = QSize(200, metrics.height())

        return self.__sizeHint

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOption()
        option.initFrom(self)

        rect = option.rect
        metrics = option.fontMetrics
        text = metrics.elidedText(self.__text, self.__textElideMode,
                                  rect.width())
        painter.drawItemText(rect, self.__alignment,
                             option.palette, self.isEnabled(), text,
                             self.foregroundRole())
        painter.end()

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.__update()

        return QWidget.changeEvent(self, event)

    def __update(self):
        self.__sizeHint = None
        self.updateGeometry()
        self.update()


class PreviewBrowser(QWidget):
    """A Preview Browser for recent/premade scheme selection.
    """
    # Emitted when the current previewed item changes
    currentIndexChanged = Signal(int)

    # Emitted when an item is double clicked in the preview list.
    activated = Signal(int)

    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.__model = None
        self.__currentIndex = -1
        self.__template = DESCRIPTION_TEMPLATE
        self.__setupUi()

    def __setupUi(self):
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(12, 12, 12, 12)

        # Top row with full text description and a large preview
        # image.
        self.__label = QLabel(self, objectName="description-label",
                              wordWrap=True,
                              alignment=Qt.AlignTop | Qt.AlignLeft)

        self.__label.setWordWrap(True)
        self.__label.setFixedSize(220, PREVIEW_SIZE[1])

        self.__image = QSvgWidget(self, objectName="preview-image")
        self.__image.setFixedSize(*PREVIEW_SIZE)

        self.__imageFrame = DropShadowFrame(self)
        self.__imageFrame.setWidget(self.__image)

        # Path text below the description and image
        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(12, 0, 12, 0)
        path_label = QLabel("<b>{0!s}</b>".format(self.tr("Path:")), self,
                            objectName="path-label")

        self.__path = TextLabel(self, objectName="path-text")

        path_layout.addWidget(path_label)
        path_layout.addWidget(self.__path)

        self.__selectAction = \
            QAction(self.tr("Select"), self,
                    objectName="select-action",
                    )

        top_layout.addWidget(self.__label, 1,
                             alignment=Qt.AlignTop | Qt.AlignLeft)
        top_layout.addWidget(self.__image, 1,
                             alignment=Qt.AlignTop | Qt.AlignRight)

        vlayout.addLayout(top_layout)
        vlayout.addLayout(path_layout)

        # An list view with small preview icons.
        self.__previewList = LinearIconView(
            objectName="preview-list-view",
            wordWrap=True
        )
        self.__previewList.doubleClicked.connect(self.__onDoubleClicked)

        vlayout.addWidget(self.__previewList)
        self.setLayout(vlayout)

    def setModel(self, model):
        """Set the item model for preview.
        """
        if self.__model != model:
            if self.__model:
                s_model = self.__previewList.selectionModel()
                s_model.selectionChanged.disconnect(self.__onSelectionChanged)
                self.__model.dataChanged.disconnect(self.__onDataChanged)

            self.__model = model
            self.__previewList.setModel(model)

            if model:
                s_model = self.__previewList.selectionModel()
                s_model.selectionChanged.connect(self.__onSelectionChanged)
                self.__model.dataChanged.connect(self.__onDataChanged)

            if model and model.rowCount():
                self.setCurrentIndex(0)

    def model(self):
        """Return the item model.
        """
        return self.__model

    def setPreviewDelegate(self, delegate):
        """Set the delegate to render the preview images.
        """
        raise NotImplementedError

    def setDescriptionTemplate(self, template):
        self.__template = template
        self.__update()

    def setCurrentIndex(self, index):
        """Set the selected preview item index.
        """
        if self.__model is not None and self.__model.rowCount():
            index = min(index, self.__model.rowCount() - 1)
            index = self.__model.index(index, 0)
            sel_model = self.__previewList.selectionModel()
            # This emits selectionChanged signal and triggers
            # __onSelectionChanged, currentIndex is updated there.
            sel_model.select(index, sel_model.ClearAndSelect)

        elif self.__currentIndex != -1:
            self.__currentIndex = -1
            self.__update()
            self.currentIndexChanged.emit(-1)

    def currentIndex(self):
        """Return the current selected index.
        """
        return self.__currentIndex

    def __onSelectionChanged(self, *args):
        """Selected item in the preview list has changed.
        Set the new description and large preview image.

        """
        rows = self.__previewList.selectedIndexes()
        if rows:
            index = rows[0]
            self.__currentIndex = index.row()
        else:
            index = QModelIndex()
            self.__currentIndex = -1

        self.__update()
        self.currentIndexChanged.emit(self.__currentIndex)

    def __onDataChanged(self, topleft, bottomRight):
        """Data changed, update the preview if current index in the changed
        range.

        """
        if self.__currentIndex <= topleft.row() and \
                self.__currentIndex >= bottomRight.row():
            self.__update()

    def __onDoubleClicked(self, index):
        """Double click on an item in the preview item list.
        """
        self.activated.emit(index.row())

    def __update(self):
        """Update the current description.
        """
        if self.__currentIndex != -1:
            index = self.model().index(self.__currentIndex, 0)
        else:
            index = QModelIndex()

        if not index.isValid():
            description = ""
            name = ""
            path = ""
            svg = NO_PREVIEW_SVG
        else:
            description = str(index.data(Qt.WhatsThisRole))
            if not description:
                description = "No description."

            description = escape(description)
            description = description.replace("\n", "<br/>")

            name = str(index.data(Qt.DisplayRole))
            if not name:
                name = "Untitled"

            name = escape(name)
            path = str(index.data(Qt.StatusTipRole))

            svg = str(index.data(previewmodel.ThumbnailSVGRole))

        desc_text = self.__template.format(description=description, name=name)

        self.__label.setText(desc_text)

        self.__path.setText(contractuser(path))

        if not svg:
            svg = NO_PREVIEW_SVG

        if svg:
            self.__image.load(QByteArray(svg.encode("utf-8")))


def contractuser(path):
    # type: (str) -> str
    """
    Inverse of `expanduser(join("~", path))`

    Return the path unmodified if not under user's home dir.

    Parameters
    ----------
    path : str

    Returns
    -------
    path : str

    Examples
    --------
    >>> contractuser(os.path.expanduser("~/hello"))
    '~/hello'
    """
    home = os.path.expanduser("~/")
    pathnorm = os.path.normcase(os.path.normpath(path))
    homenorm = os.path.normcase(os.path.normpath(home))
    if pathnorm.startswith(homenorm):
        path = os.path.join("~", os.path.relpath(path, home))
    return path
