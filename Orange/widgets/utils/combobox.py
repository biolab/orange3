from AnyQt.QtCore import Qt, Signal
from AnyQt.QtGui import (
    QBrush, QColor, QPalette, QPen, QFont, QFontMetrics, QFocusEvent
)
from AnyQt.QtWidgets import (
    QStylePainter, QStyleOptionComboBox, QStyle, QApplication, QLineEdit
)

from orangewidget.utils.combobox import (
    ComboBoxSearch, ComboBox, qcombobox_emit_activated
)

__all__ = [
    "ComboBoxSearch", "ComboBox", "ItemStyledComboBox", "TextEditCombo"
]


class ItemStyledComboBox(ComboBox):
    """
    A QComboBox that draws its text using current item's foreground and font
    role.

    Note
    ----
    Stylesheets etc. can completely ignore this.
    """
    def __init__(self, *args, placeholderText="", **kwargs):
        self.__placeholderText = placeholderText
        super().__init__(*args, **kwargs)

    def paintEvent(self, _event) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        foreground = self.currentData(Qt.ForegroundRole)
        if isinstance(foreground, (QBrush, QColor)):
            foreground = QBrush(foreground)
            if foreground.style() != Qt.NoBrush:
                # some styles take WindowText some just use current pen?
                option.palette.setBrush(QPalette.WindowText, foreground)
                option.palette.setBrush(QPalette.ButtonText, foreground)
                option.palette.setBrush(QPalette.Text, foreground)
                painter.setPen(QPen(foreground, painter.pen().widthF()))
        font = self.currentData(Qt.FontRole)
        if isinstance(font, QFont):
            option.fontMetrics = QFontMetrics(font)
            painter.setFont(font)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)

    def placeholderText(self) -> str:
        """
        Return the placeholder text.

        Returns
        -------
        text : str
        """
        return self.__placeholderText

    def setPlaceholderText(self, text: str):
        """
        Set the placeholder text.

        This text is displayed on the checkbox when the currentIndex() == -1

        Parameters
        ----------
        text : str
        """
        if self.__placeholderText != text:
            self.__placeholderText = text
            self.update()

    def initStyleOption(self, option: 'QStyleOptionComboBox') -> None:
        super().initStyleOption(option)
        if self.currentIndex() == -1:
            option.currentText = self.__placeholderText
            option.palette.setCurrentColorGroup(QPalette.Disabled)


class TextEditCombo(ComboBox):
    #: This signal is emitted whenever the contents of the combo box are
    #: changed and the widget loses focus *OR* via item activation (activated
    #: signal)
    editingFinished = Signal()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("editable", True)
        # `activated=...` kwarg needs to be connected after `__on_activated`
        activated = kwargs.pop("activated", None)
        self.__edited = False
        super().__init__(*args, **kwargs)
        self.activated.connect(self.__on_activated)
        if activated is not None:
            self.activated.connect(activated)
        ledit = self.lineEdit()
        if ledit is not None:
            ledit.textEdited.connect(self.__markEdited)

    def setLineEdit(self, edit: QLineEdit) -> None:
        super().setLineEdit(edit)
        edit.textEdited.connect(self.__markEdited)

    def __markEdited(self):
        self.__edited = True

    def __on_activated(self):
        self.__edited = False  # mark clean on any activation
        self.editingFinished.emit()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        popup = QApplication.activePopupWidget()
        if self.isEditable() and self.__edited and \
                (event.reason() != Qt.PopupFocusReason or
                    not (popup is not None
                         and popup.parent() in (self, self.lineEdit()))):
            def monitor():
                # monitor if editingFinished was emitted from
                # __on_editingFinished to avoid double emit.
                nonlocal emitted
                emitted = True
            emitted = False
            self.editingFinished.connect(monitor)
            self.__edited = False
            self.__on_editingFinished()
            self.editingFinished.disconnect(monitor)

            if not emitted:
                self.editingFinished.emit()

    def __on_editingFinished(self):
        le = self.lineEdit()
        policy = self.insertPolicy()
        text = le.text()
        if not text:
            return
        index = self.findText(text, Qt.MatchFixedString)
        if index != -1:
            self.setCurrentIndex(index)
            qcombobox_emit_activated(self, index)
            return
        if policy == ComboBox.NoInsert:
            return
        elif policy == ComboBox.InsertAtTop:
            index = 0
        elif policy == ComboBox.InsertAtBottom:
            index = self.count()
        elif policy == ComboBox.InsertAfterCurrent:
            index = self.currentIndex() + 1
        elif policy == ComboBox.InsertBeforeCurrent:
            index = max(self.currentIndex(), 0)
        elif policy == ComboBox.InsertAlphabetically:
            for index in range(self.count()):
                if self.itemText(index).lower() >= text.lower():
                    break
        elif policy == ComboBox.InsertAtCurrent:
            self.setItemText(self.currentIndex(), text)
            qcombobox_emit_activated(self, self.currentIndex())
            return

        if index > -1:
            self.insertItem(index, text)
            self.setCurrentIndex(index)
            qcombobox_emit_activated(self, self.currentIndex())

    def text(self):
        # type: () -> str
        """
        Return the current text.
        """
        return self.itemText(self.currentIndex())

    def setText(self, text):
        # type: (str) -> None
        """
        Set `text` as the current text (adding it to the model if necessary).
        """
        idx = self.findData(text, Qt.EditRole, Qt.MatchExactly)
        if idx != -1:
            self.setCurrentIndex(idx)
        else:
            self.addItem(text)
            self.setCurrentIndex(self.count() - 1)
