"""
A dialog widget for selecting an item.
"""

from AnyQt.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QDialogButtonBox, QSizePolicy
)
from AnyQt.QtCore import Qt, QStringListModel
from AnyQt.QtCore import pyqtSignal as Signal

from . import previewbrowser


class PreviewDialog(QDialog):
    """A Dialog for selecting an item from a PreviewItem.
    """
    currentIndexChanged = Signal(int)

    def __init__(self, parent=None, flags=Qt.WindowFlags(0),
                 model=None, **kwargs):
        QDialog.__init__(self, parent, flags, **kwargs)

        self.__setupUi()
        if model is not None:
            self.setModel(model)

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)

        self.__browser = previewbrowser.PreviewBrowser(
            self, heading="<h3>{0}</h3>".format(self.tr("Preview"))
        )
        self.__buttons = QDialogButtonBox(QDialogButtonBox.Open |
                                          QDialogButtonBox.Cancel,
                                          Qt.Horizontal,)
        self.__buttons.button(QDialogButtonBox.Open).setAutoDefault(True)

        # Set the Open dialog as disabled until the current index changes
        self.__buttons.button(QDialogButtonBox.Open).setEnabled(False)

        # The QDialogButtonsWidget messes with the layout if it is
        # contained directly in the QDialog. So we create an extra
        # layer of indirection.
        buttons = QWidget(objectName="button-container")
        buttons_l = QVBoxLayout()
        buttons_l.setContentsMargins(12, 0, 12, 12)
        buttons.setLayout(buttons_l)

        buttons_l.addWidget(self.__buttons)

        layout.addWidget(self.__browser)

        layout.addWidget(buttons)

        self.__buttons.accepted.connect(self.accept)
        self.__buttons.rejected.connect(self.reject)
        self.__browser.currentIndexChanged.connect(
            self.__on_currentIndexChanged
        )
        self.__browser.activated.connect(self.__on_activated)

        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setItems(self, items):
        """Set the items (a list of strings) for preview/selection.
        """
        model = QStringListModel(items)
        self.setModel(model)

    def setModel(self, model):
        """Set the model for preview/selection.
        """
        self.__browser.setModel(model)

    def model(self):
        """Return the model.
        """
        return self.__browser.model()

    def currentIndex(self):
        return self.__browser.currentIndex()

    def setCurrentIndex(self, index):
        """Set the current selected (shown) index.
        """
        self.__browser.setCurrentIndex(index)

    def setHeading(self, heading):
        """Set `heading` as the heading string ('<h3>Preview</h3>'
        by default).

        """
        self.__browser.setHeading(heading)

    def heading(self):
        """Return the heading string.
        """
    def __on_currentIndexChanged(self, index):
        button = self.__buttons.button(QDialogButtonBox.Open)
        button.setEnabled(index >= 0)
        self.currentIndexChanged.emit(index)

    def __on_activated(self, index):
        if self.currentIndex() != index:
            self.setCurrentIndex(index)

        self.accept()
