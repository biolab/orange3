"""
Orange Canvas Welcome Dialog

"""

from PyQt4.QtGui import (
    QDialog, QWidget, QToolButton, QCheckBox, QAction,
    QHBoxLayout, QVBoxLayout, QFont, QSizePolicy,
    QPixmap, QIcon, QPainter, QColor, QBrush
)

from PyQt4.QtCore import Qt, QRect, QPoint
from PyQt4.QtCore import pyqtSignal as Signal

from ..canvas.items.utils import radial_gradient
from ..registry import NAMED_COLORS


def decorate_welcome_icon(icon, background_color):
    """Return a `QIcon` with a circle shaped background.
    """
    welcome_icon = QIcon()
    sizes = [32, 48, 64, 80]
    background_color = NAMED_COLORS.get(background_color, background_color)
    background_color = QColor(background_color)
    grad = radial_gradient(background_color)
    for size in sizes:
        icon_pixmap = icon.pixmap(5 * size / 8, 5 * size / 8)
        icon_size = icon_pixmap.size()
        icon_rect = QRect(QPoint(0, 0), icon_size)

        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        ellipse_rect = QRect(0, 0, size, size)
        p.drawEllipse(ellipse_rect)
        icon_rect.moveCenter(ellipse_rect.center())
        p.drawPixmap(icon_rect.topLeft(), icon_pixmap)
        p.end()

        welcome_icon.addPixmap(pixmap)

    return welcome_icon


WELCOME_WIDGET_BUTTON_STYLE = \
"""

WelcomeActionButton {
    border: none;
    icon-size: 75px;
    /*font: bold italic 14px "Helvetica";*/
}

WelcomeActionButton:pressed {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #dadbde, stop: 1 #f6f7fa);
    border-radius: 10px;
}

WelcomeActionButton:focus {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #dadbde, stop: 1 #f6f7fa);
    border-radius: 10px;
}

"""


class WelcomeActionButton(QToolButton):
    def __init__(self, parent=None):
        QToolButton.__init__(self, parent)

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)


class WelcomeDialog(QDialog):
    """A welcome widget shown at startup presenting a series
    of buttons (actions) for a beginner to choose from.

    """
    triggered = Signal(QAction)

    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)

        self.__triggeredAction = None

        self.setupUi()

    def setupUi(self):
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.__mainLayout = QVBoxLayout()
        self.__mainLayout.setContentsMargins(0, 40, 0, 40)
        self.__mainLayout.setSpacing(65)

        self.layout().addLayout(self.__mainLayout)

        self.setStyleSheet(WELCOME_WIDGET_BUTTON_STYLE)

        bottom_bar = QWidget(objectName="bottom-bar")
        bottom_bar_layout = QHBoxLayout()
        bottom_bar_layout.setContentsMargins(20, 10, 20, 10)
        bottom_bar.setLayout(bottom_bar_layout)
        bottom_bar.setSizePolicy(QSizePolicy.MinimumExpanding,
                                 QSizePolicy.Maximum)

        check = QCheckBox(self.tr("Dont'show again at startup"), bottom_bar)
        check.setChecked(False)

        self.__showAtStartupCheck = check

        bottom_bar_layout.addWidget(check, alignment=Qt.AlignVCenter | \
                                    Qt.AlignLeft)

        self.layout().addWidget(bottom_bar, alignment=Qt.AlignBottom,
                                stretch=1)

        self.setSizeGripEnabled(False)
        self.setFixedSize(620, 390)

    def setShowAtStartup(self, show):
        if self.__showAtStartupCheck.isChecked() != (not show):
            self.__showAtStartupCheck.setChecked(not show)

    def showAtStartup(self):
        return not self.__showAtStartupCheck.isChecked()

    def addRow(self, actions, background="light-orange"):
        """Add a row with `actions`.
        """
        count = self.__mainLayout.count()
        self.insertRow(count, actions, background)

    def insertRow(self, index, actions, background="light-orange"):
        """Insert a row with `actions` at `index`.
        """
        widget = QWidget(objectName="icon-row")
        layout = QHBoxLayout()
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(65)
        widget.setLayout(layout)

        self.__mainLayout.insertWidget(index, widget, stretch=10,
                                       alignment=Qt.AlignCenter)

        for i, action in enumerate(actions):
            self.insertAction(index, i, action, background)

    def insertAction(self, row, index, action,
                      background="light-orange"):
        """Insert `action` in `row` in position `index`.
        """
        button = self.createButton(action, background)
        self.insertButton(row, index, button)

    def insertButton(self, row, index, button):
        """Insert `button` in `row` in position `index`.
        """
        item = self.__mainLayout.itemAt(row)
        layout = item.widget().layout()
        layout.insertWidget(index, button)
        button.triggered.connect(self.__on_actionTriggered)

    def createButton(self, action, background="light-orange"):
        """Create a tool button for action.
        """
        button = WelcomeActionButton(self)
        button.setDefaultAction(action)
        button.setText(action.iconText())
        button.setIcon(decorate_welcome_icon(action.icon(), background))
        button.setToolTip(action.toolTip())
        button.setFixedSize(100, 100)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        font = QFont(button.font())
        font.setPointSize(13)
        button.setFont(font)

        return button

    def buttonAt(self, i, j):
        """Return the button at i-t row and j-th column.
        """
        item = self.__mainLayout.itemAt(i)
        row = item.widget()
        item = row.layout().itemAt(j)
        return item.widget()

    def triggeredAction(self):
        """Return the action that was triggered by the user.
        """
        return self.__triggeredAction

    def showEvent(self, event):
        # Clear the triggered action before show.
        self.__triggeredAction = None
        QDialog.showEvent(self, event)

    def __on_actionTriggered(self, action):
        """Called when the button action is triggered.
        """
        self.triggered.emit(action)
        self.__triggeredAction = action
