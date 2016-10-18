from PyQt4.QtGui import QPushButton, QStyle, QStyleOptionButton
from PyQt4.QtCore import Qt, QSize


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
