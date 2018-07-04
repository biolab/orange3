from AnyQt.QtWidgets import (
    QPushButton, QAbstractButton, QFocusFrame, QStyle, QStylePainter,
    QStyleOptionButton
)
from AnyQt.QtGui import QPalette, QIcon
from AnyQt.QtCore import Qt, QSize, QEvent


class VariableTextPushButton(QPushButton):
    """
    QPushButton subclass with an sizeHint method to better support settable
    variable width button text.

    Use this class instead of the QPushButton when the button will
    switch the text dynamically while displayed.
    """
    def __init__(self, *args, textChoiceList=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.__textChoiceList = list(textChoiceList)

    def setTextChoiceList(self, textList):
        """
        Set the list of all `text` string to use for size hinting.

        Parameters
        ----------
        textList : List[str]
            A list of all different `text` properties that will/can be set on
            the push button. This list is used to derive a suitable sizeHint
            for the widget.
        """
        self.__textChoiceList = textList
        self.updateGeometry()

    def sizeHint(self):
        """
        Reimplemented from `QPushButton.sizeHint`.

        Returns
        -------
        sh : QSize
        """
        sh = super().sizeHint()
        option = QStyleOptionButton()
        self.initStyleOption(option)
        style = self.style()
        fm = option.fontMetrics
        if option.iconSize.isValid():
            icsize = option.iconSize
            icsize.setWidth(icsize.width() + 4)
        else:
            icsize = QSize()

        for text in self.__textChoiceList:
            option.text = text
            size = fm.size(Qt.TextShowMnemonic, text)

            if not icsize.isNull():
                size.setWidth(size.width() + icsize.width())
                size.setHeight(max(size.height(), icsize.height()))

            sh = sh.expandedTo(
                style.sizeFromContents(QStyle.CT_PushButton, option,
                                       size, self))
        return sh


class SimpleButton(QAbstractButton):
    """
    A simple icon button widget.
    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__focusframe = None

    def focusInEvent(self, event):
        # reimplemented
        event.accept()
        if self.__focusframe is None:
            self.__focusframe = QFocusFrame(self)
            self.__focusframe.setWidget(self)
            palette = self.palette()
            palette.setColor(QPalette.Foreground,
                             palette.color(QPalette.Highlight))
            self.__focusframe.setPalette(palette)

    def focusOutEvent(self, event):
        # reimplemented
        event.accept()
        if self.__focusframe is not None:
            self.__focusframe.hide()
            self.__focusframe.deleteLater()
            self.__focusframe = None

    def event(self, event):
        if event.type() == QEvent.Enter or event.type() == QEvent.Leave:
            self.update()
        return super().event(event)

    def sizeHint(self):
        # reimplemented
        self.ensurePolished()
        iconsize = self.iconSize()
        icon = self.icon()
        if not icon.isNull():
            iconsize = icon.actualSize(iconsize)
        return iconsize

    def minimumSizeHint(self):
        # reimplemented
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        option.initFrom(self)
        option.text = ""
        option.icon = self.icon()
        option.iconSize = self.iconSize()
        option.features = QStyleOptionButton.Flat
        if self.isDown():
            option.state |= QStyle.State_Sunken
            painter.drawPrimitive(QStyle.PE_PanelButtonBevel, option)

        if not option.icon.isNull():
            if option.state & QStyle.State_Active:
                mode = (QIcon.Active if option.state & QStyle.State_MouseOver
                        else QIcon.Normal)
            else:
                mode = QIcon.Disabled
            if self.isChecked():
                state = QIcon.On
            else:
                state = QIcon.Off
            option.icon.paint(painter, option.rect, Qt.AlignCenter, mode, state)
