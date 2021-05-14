import sys
import codecs

from typing import List, Iterable

from AnyQt.QtCore import Qt, QModelIndex, QAbstractItemModel, QSettings
from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QListView, QDialogButtonBox, QSizePolicy,
    QStyle, QStyleOption,
)
from AnyQt.QtCore import pyqtSlot as Slot


DEFAULT_ENCODINGS = [
    "utf-8", "utf-16", "utf-32",
    "iso8859-1",  # latin 1
    "shift_jis", "iso2022_jp",
    "gb18030",
    "euc_kr",
]

ENCODING_DISPLAY_NAME = (
    # pylint: disable=bad-whitespace
    ("utf-8",           "Unicode (UTF-8)"),
    ("utf-16",          "Unicode (UTF-16)"),
    ("utf-16-le",       "Unicode (UTF-16LE)"),
    ("utf-16-be",       "Unicode (UTF-16BE)"),
    ("utf-32",          "Unicode (UTF-32)"),
    ("utf-32-le",       "Unicode (UTF-32LE)"),
    ("utf-32-be",       "Unicode (UTF-32BE)"),
    ("utf-7",           "Unicode (UTF-7)"),
    ("ascii",           "English (US-ASCII)"),

    ("iso8859-1",       "Western Europe (ISO Latin 1)"),
    ("iso8859-15",      "Western Europe (ISO-8859-15)"),
    ("cp1252",          "Western Europe (Windows-1252)"),
    ("mac_roman",       "Western Europe (Mac OS Roman)"),

    ("iso8859-2",       "Central and Eastern Europe (ISO Latin 2)"),
    ("cp1250",          "Central and Eastern Europe (Windows-1250)"),
    ("mac_latin2",      "Central and Eastern Europe (Mac Latin-2)"),

    ("iso8859-3",       "Esperanto, Maltese (ISO Latin 3)"),
    ("iso8859-4",       "Baltic Languages (ISO Latin 4)"),
    ("cp1257",          "Baltic Languages (Windows-1257)"),
    ("iso8859-13",      "Baltic Languages (ISO-8859-13)"),
    ("iso8859-16",      "South-Eastern Europe (ISO-8859-16)"),

    ("iso8859-5",       "Cyrillic (ISO-8859-5)"),
    ("cp1251",          "Cyrillic (Windows-1251)"),
    ("mac_cyrillic",    "Cyrillic (Mac OS Cyrillic)"),
    ("koi8-r",          "Cyrillic (KOI8-R)"),
    ("koi8-u",          "Cyrillic (KOI8-U)"),

    ("iso8859-14",      "Celtic Languages (ISO-8859-14)"),
    ("iso8859-10",      "Nordic Languages (ISO-8859-10)"),
    ("mac_iceland",     "Icelandic (Mac Iceland)"),

    ("iso8859-7",       "Greek (ISO-8859-7)"),
    ("cp1253",          "Greek (Windows-1253)"),
    ("mac_greek",       "Greek (Mac Greek)"),

    ("iso8859-8",       "Hebrew (ISO-8859-8)"),
    ("cp1255",          "Hebrew (Windows-1255)"),

    ("iso8859-6",       "Arabic (ISO-8859-6)"),
    ("cp1256",          "Arabic (Windows-1256)"),

    ("iso8859-9",       "Turkish (ISO-8859-9)"),
    ("cp1254",          "Turkish (Windows-1254)"),
    ("mac_turkish",     "Turkish (Mac Turkish)"),

    ("iso8859-11",      "Thai (ISO-8859-11)"),

    ("iso2022_jp",      "Japanese (ISO-2022-JP)"),
    ("iso2022_jp_1",    "Japanese (ISO-2022-JP-1)"),
    ("iso2022_jp_2",    "Japanese (ISO-2022-JP-2)"),
    ("iso2022_jp_2004", "Japanese (ISO-2022-JP-2004)"),
    ("iso2022_jp_3",    "Japanese (ISO-2022-JP-3)"),
    ("shift_jis",       "Japanese (Shift JIS)"),
    ("shift_jis_2004",  "Japanese (Shift JIS 2004)"),
    ("euc_jp",          "Japanese (EUC-JP)"),

    ("iso2022_kr",      "Korean (ISO-2022-KR)"),
    ("euc_kr",          "Korean (EUC-KR)"),

    ("gb2312",          "Simplified Chinese (GB 2312)"),
    ("gbk",             "Chinese (GBK)"),
    ("gb18030",         "Chinese (GB 18030)"),
    ("big5",            "Traditional Chinese (BIG5)"),
    ("big5hkscs",       "Traditional Chinese (BIG5-HKSC)"),

    ("cp1258",          "Vietnamese (Windows-1258)"),
    ("koi8-t",          "Tajik (KOI8-T)"),
)

ENCODINGS = [code for code, _ in ENCODING_DISPLAY_NAME]

__display_name = None


def display_name(codec):
    # type: (str) -> str
    """
    Return a human readable display name for a codec if available

    Parameters
    ----------
    codec : str
        A codec name (as accepted by `codecs.lookup`).

    Returns
    -------
    name : str
    """
    global __display_name
    if __display_name is None:
        d = {}
        for k, name in ENCODING_DISPLAY_NAME:
            try:
                co = codecs.lookup(k)
            except LookupError:
                pass
            else:
                d[co.name] = name
        __display_name = d
    try:
        co = codecs.lookup(codec)
    except LookupError:
        return codec
    else:
        return __display_name.get(co.name, codec)


class EncodingsView(QListView):
    """
    QListView with size hinting based on contents.
    """
    def sizeHint(self):
        self.ensurePolished()
        sh = super().sizeHint()
        frame = self.frameWidth()
        style = self.style()
        s = self.sizeHintForColumn(0)
        m = self.viewportMargins()
        opt = QStyleOption()
        opt.initFrom(self)

        spacing = style.pixelMetric(
            QStyle.PM_ScrollView_ScrollBarSpacing, opt, self)
        extent = style.pixelMetric(
            QStyle.PM_ScrollBarExtent, opt, self)
        overlap = style.pixelMetric(
            QStyle.PM_ScrollView_ScrollBarOverlap, opt, self)
        width = (s + extent - overlap + 2 * spacing + 2 * frame +
                 m.left() + m.right())
        sh.setWidth(max(sh.width(), width,))
        return sh

    def dataChanged(self, topLeft, bottomRight, roles=()):
        super().dataChanged(topLeft, bottomRight, roles)
        self.updateGeometry()


class SelectEncodingsWidget(QWidget):
    """
    Popup widget for selecting a set of text encodings for use in other parts
    of the GUI.
    """
    def __init__(self, *args, headingText="", **kwargs):
        super().__init__(*args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.__top_label = QLabel(
            headingText, visible=bool(headingText),
            objectName="-top-heading-text"
        )
        self.__top_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.__model = model = encodings_model()
        self.__model.setParent(self)

        self.__view = view = EncodingsView(
            self, uniformItemSizes=True, editTriggers=QListView.NoEditTriggers
        )
        self.__view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        view.setModel(model)

        self.layout().addWidget(self.__top_label)
        self.layout().addWidget(view)

        buttons = QDialogButtonBox(
            standardButtons=QDialogButtonBox.RestoreDefaults
        )
        b = buttons.addButton("Select all", QDialogButtonBox.ActionRole)
        b.clicked.connect(self.selectAll)
        b = buttons.button(QDialogButtonBox.RestoreDefaults)
        b.clicked.connect(self.reset)
        self.layout().addWidget(buttons)
        self.setAttribute(Qt.WA_MacSmallSize)

    def headingText(self):
        # type: () -> str
        return self.__top_label.text()

    def setHeadingText(self, text):
        # type: (str) -> None
        self.__top_label.setText(text)
        self.__top_label.setVisible(bool(text))

    def setModel(self, model):
        if self.__model is not None:
            if self.__model.parent() is self:
                self.__model.deleteLater()
            self.__model = None

        self.__view.setModel(model)
        self.__model = model

    def model(self):
        return self.__model

    def selectedEncodings(self):
        # type: () -> List[str]
        """
        Return a list of currently selected (checked) encodings.
        """
        model = self.__model
        res = []
        for i in range(model.rowCount()):
            data = model.itemData(model.index(i, 0))
            if data.get(Qt.CheckStateRole) == Qt.Checked and \
                    EncodingNameRole in data:
                res.append(data[EncodingNameRole])
        return res

    @Slot()
    def selectAll(self):
        """
        Select all encodings.
        """
        model = self.__model
        for i in range(model.rowCount()):
            item = model.item(i)
            item.setCheckState(Qt.Checked)

    @Slot()
    def clearAll(self):
        """
        Clear (uncheck) all encodings.
        """
        model = self.__model
        for i in range(model.rowCount()):
            item = model.item(i)
            item.setCheckState(Qt.Checked)

    @Slot()
    def reset(self):
        """
        Reset the encodings model to the default selected set.
        """
        model = self.__model
        for i in range(model.rowCount()):
            item = model.item(i)
            co = item.data(CodecInfoRole)
            if isinstance(co, codecs.CodecInfo):
                state = co.name in DEFAULT_ENCODINGS
                item.setCheckState(
                    Qt.Checked if state else Qt.Unchecked
                )


SettingsGroup = __name__ + "#selected-text-encodings"


def list_selected_encodings():
    # type: () -> List[str]
    """
    Return a list of all current selected encodings from user preferences.
    """
    settings = QSettings()
    settings.beginGroup(SettingsGroup)
    res = []
    for encoding, _ in ENCODING_DISPLAY_NAME:
        try:
            co = codecs.lookup(encoding)
        except LookupError:
            continue
        selected = settings.value(
            co.name, defaultValue=co.name in DEFAULT_ENCODINGS, type=bool
        )
        if selected:
            res.append(co.name)
    return res


EncodingNameRole = Qt.UserRole
CodecInfoRole = Qt.UserRole + 0xC1


def encodings_model():
    # type: () -> QAbstractItemModel
    """
    Return a list model of text encodings.

    The items are checkable and initialized based on current stored user
    preferences. Any change in check state is stored and writen back
    immediately.

    The normalized encoding (codec) names are accessible using `Qt.UserRole`

    Returns
    -------
    model : QAbstractItemModel
    """
    m = QStandardItemModel()
    items = []
    settings = QSettings()
    settings.beginGroup(SettingsGroup)

    def is_selected(co):
        # type: (codecs.CodecInfo) -> bool
        return settings.value(
            co.name, defaultValue=co.name in DEFAULT_ENCODINGS, type=bool)

    def store_selected(index):
        # type: (QModelIndex) -> None
        # write back the selected state for index
        co = index.data(CodecInfoRole)
        state = index.data(Qt.CheckStateRole)
        if isinstance(co, codecs.CodecInfo):
            settings.setValue(co.name, state == Qt.Checked)

    for encoding, name in ENCODING_DISPLAY_NAME:
        try:
            co = codecs.lookup(encoding)
        except LookupError:
            continue

        item = QStandardItem(name)
        item.setData(co.name, EncodingNameRole)
        item.setData(co, CodecInfoRole)
        item.setToolTip(name + "; " + encoding)
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled |
                      Qt.ItemIsSelectable)
        item.setCheckState(Qt.Checked if is_selected(co) else Qt.Unchecked)
        items.append(item)

    def on_data_changed(first, last, roles=()):
        # type: (QModelIndex, QModelIndex, Iterable[int]) -> None
        if roles and Qt.CheckStateRole not in roles:
            return
        assert first.column() == last.column()
        for i in range(first.row(), last.row() + 1):
            index = first.sibling(i, first.column())
            store_selected(index)

    m.invisibleRootItem().appendRows(items)
    m.dataChanged.connect(on_data_changed)
    return m


def main(args=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(args)
    w = SelectEncodingsWidget(
        headingText="Select encodings visible in text encoding menus"
    )
    w.show()
    w.activateWindow()
    app.exec()


if __name__ == "__main__":
    sys.exit(main())
