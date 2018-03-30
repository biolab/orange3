from AnyQt.QtWidgets import (
    QListView, QSizePolicy, QStyle, QStyleOptionViewItem
)
from AnyQt.QtCore import Qt, QSize


class LinearIconView(QListView):
    """
    An list view (in QListView.IconMode) with no item wrapping.

    Suitable for displaying large(ish) icons with text in a single row/column.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setViewMode(QListView.IconMode)
        self.setWrapping(False)
        self.setWordWrap(True)

        self.setSelectionMode(QListView.SingleSelection)
        self.setEditTriggers(QListView.NoEditTriggers)
        self.setMovement(QListView.Static)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding,
                           QSizePolicy.Fixed)

        self.setIconSize(QSize(120, 80))

    def sizeHint(self):
        # type: () -> QSize
        """
        Reimplemented.

        Provide sensible size hint based on the view's contents.
        """
        flow = self.flow()

        if self.model() is None or not self.model().rowCount():
            style = self.style()
            opt = self.viewOptions()
            opt.features |= (QStyleOptionViewItem.HasDecoration |
                             QStyleOptionViewItem.HasDisplay |
                             QStyleOptionViewItem.WrapText)
            opt.text = "X" * 12 + "\nX"
            sh = style.sizeFromContents(
                QStyle.CT_ItemViewItem, opt, QSize(), self)
        else:
            # Sample the first 20 items for a size hint. The objective is to
            # get a representative height due to the word wrapping
            model = self.model()
            samplesize = min(20, model.rowCount())
            shs = [self.sizeHintForIndex(model.index(i, 0))
                   for i in range(samplesize)]
            if flow == QListView.TopToBottom:
                sh = QSize(max(s.width() for s in shs), 200)
            else:
                sh = QSize(200, max(s.height() for s in shs))

        left, top, right, bottom = self.getContentsMargins()
        if flow == QListView.TopToBottom:
            sh = sh + QSize(left + right, 0)
        else:
            sh = sh + QSize(0, top + bottom)

        if flow == QListView.TopToBottom and \
                self.verticalScrollBarPolicy() != Qt.ScrollBarAlwaysOff:
            ssh = self.verticalScrollBar().sizeHint()
            return QSize(sh.width() + ssh.width(), sh.height())
        elif self.flow() == QListView.LeftToRight and \
                self.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff:
            ssh = self.horizontalScrollBar().sizeHint()
            return QSize(sh.width(), sh.height() + ssh.height())
        else:
            return sh

    def updateGeometries(self):
        """Reimplemented"""
        super().updateGeometries()
        self.updateGeometry()

    def dataChanged(self, topLeft, bottomRight, roles=[]):
        """Reimplemented"""
        super().dataChanged(topLeft, bottomRight)
        self.updateGeometry()
