import sys
from typing import Tuple, List, Dict, Iterable

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication
from AnyQt.QtGui import QFont, QFontDatabase

import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample

from orangewidget.utils.visual_settings_dlg import KeyType, ValueType, \
    SettingsType, FontList

_SettingType = Dict[str, ValueType]
_LegendItemType = Tuple[ItemSample, pg.LabelItem]


def available_font_families() -> List:
    """
    Function returns list of available font families.
    Can be used to instantiate font combo boxes.

    Returns
    -------
    fonts: list
        List of available font families.
    """
    if not QApplication.instance():
        _ = QApplication(sys.argv)
    fonts = QFontDatabase().families()
    default = default_font_family()

    defaults = [default]
    if default in fonts:
        fonts.remove(default)

    guessed_name = default.split()[0]
    i = 0
    while i < len(fonts):
        if fonts[i].startswith(guessed_name):
            defaults.append(fonts.pop(i))
        else:
            i += 1
    return FontList(defaults
                    + [""]
                    + sorted(fonts, key=lambda s: s.replace(".", "")))


def default_font_family() -> str:
    """
    Function returns default font family used in Qt application.
    Can be used to instantiate initial dialog state.

    Returns
    -------
    font: str
        Default font family.
    """
    if not QApplication.instance():
        _ = QApplication(sys.argv)
    return QFont().family()


def default_font_size() -> int:
    """
    Function returns default font size in points used in Qt application.
    Can be used to instantiate initial dialog state.

    Returns
    -------
    size: int
        Default font size in points.
    """
    if not QApplication.instance():
        _ = QApplication(sys.argv)
    return QFont().pointSize()


class Updater:
    """ Class with helper functions and constants. """
    FONT_FAMILY_LABEL, SIZE_LABEL, IS_ITALIC_LABEL = \
        "Font family", "Font size", "Italic"

    WIDTH_LABEL, ALPHA_LABEL, STYLE_LABEL, ANTIALIAS_LABEL = \
        "Width", "Opacity", "Style", "Antialias"
    LINE_STYLES = {"Solid line": Qt.SolidLine,
                   "Dash line": Qt.DashLine,
                   "Dot line": Qt.DotLine,
                   "Dash dot line": Qt.DashDotLine,
                   "Dash dot dot line": Qt.DashDotDotLine}
    DEFAULT_LINE_STYLE = "Solid line"

    @staticmethod
    def update_plot_title_text(title_item: pg.LabelItem, text: str):
        title_item.text = text
        title_item.setVisible(bool(text))
        title_item.item.setPlainText(text)
        Updater.plot_title_resize(title_item)

    @staticmethod
    def update_plot_title_font(title_item: pg.LabelItem,
                               **settings: _SettingType):
        font = Updater.change_font(title_item.item.font(), settings)
        title_item.item.setFont(font)
        title_item.item.setPlainText(title_item.text)
        Updater.plot_title_resize(title_item)

    @staticmethod
    def plot_title_resize(title_item):
        height = title_item.item.boundingRect().height() + 6 \
            if title_item.text else 0
        title_item.setMaximumHeight(height)
        title_item.parentItem().layout.setRowFixedHeight(0, height)
        title_item.resizeEvent(None)

    @staticmethod
    def update_axis_title_text(item: pg.AxisItem, text: str):
        item.setLabel(text, item.labelUnits, item.labelUnitPrefix)
        item.resizeEvent(None)

    @staticmethod
    def update_axes_titles_font(items: List[pg.AxisItem],
                                **settings: _SettingType):
        for item in items:
            font = Updater.change_font(item.label.font(), settings)
            item.label.setFont(font)
            fstyle = ["normal", "italic"][font.italic()]
            style = {"font-size": f"{font.pointSize()}pt",
                     "font-family": f"{font.family()}",
                     "font-style": f"{fstyle}"}
            item.setLabel(item.labelText, item.labelUnits,
                          item.labelUnitPrefix, **style)

    @staticmethod
    def update_axes_ticks_font(items: List[pg.AxisItem],
                               **settings: _SettingType):
        for item in items:
            font = item.style["tickFont"] or QFont()
            # remove when contained in setTickFont() - version 0.11.0
            item.style['tickFont'] = font
            item.setTickFont(Updater.change_font(font, settings))

    @staticmethod
    def update_legend_font(items: Iterable[_LegendItemType],
                           **settings: _SettingType):
        for sample, label in items:
            if "size" in label.opts:
                # pyqtgraph added html-like support for size in 0.11.1, which
                # overrides our QFont property
                label.opts.pop("size")
                label.setText(label.text)
            sample.setFixedHeight(sample.height())
            sample.setFixedWidth(sample.width())
            label.item.setFont(Updater.change_font(label.item.font(), settings))
            bounds = label.itemRect()
            label.setMaximumWidth(bounds.width())
            label.setMaximumHeight(bounds.height())
            label.updateMin()
            label.resizeEvent(None)
            label.updateGeometry()

    @staticmethod
    def update_num_legend_font(legend: pg.LegendItem,
                               **settings: _SettingType):
        if not legend:
            return
        for sample, _ in legend.items:
            sample.set_font(Updater.change_font(sample.font, settings))
            legend.setGeometry(sample.boundingRect())

    @staticmethod
    def update_label_font(items: List[pg.TextItem], font: QFont):
        for item in items:
            item.setFont(font)

    @staticmethod
    def change_font(font: QFont, settings: _SettingType) -> QFont:
        assert all(s in (Updater.FONT_FAMILY_LABEL, Updater.SIZE_LABEL,
                         Updater.IS_ITALIC_LABEL) for s in settings), settings

        family = settings.get(Updater.FONT_FAMILY_LABEL)
        if family is not None:
            font.setFamily(family)
        size = settings.get(Updater.SIZE_LABEL)
        if size is not None:
            font.setPointSize(size)
        italic = settings.get(Updater.IS_ITALIC_LABEL)
        if italic is not None:
            font.setItalic(italic)
        return font

    @staticmethod
    def update_lines(items: List[pg.PlotCurveItem], **settings: _SettingType):
        for item in items:
            antialias = settings.get(Updater.ANTIALIAS_LABEL)
            if antialias is not None:
                item.setData(item.xData, item.yData, antialias=antialias)

            pen = item.opts["pen"]
            alpha = settings.get(Updater.ALPHA_LABEL)
            if alpha is not None:
                color = pen.color()
                color.setAlpha(alpha)
                pen.setColor(color)

            style = settings.get(Updater.STYLE_LABEL)
            if style is not None:
                pen.setStyle(Updater.LINE_STYLES[style])

            width = settings.get(Updater.WIDTH_LABEL)
            if width is not None:
                pen.setWidth(width)

            item.setPen(pen)

    @staticmethod
    def update_inf_lines(items, **settings):
        for item in items:
            pen = item.pen

            alpha = settings.get(Updater.ALPHA_LABEL)
            if alpha is not None:
                color = pen.color()
                color.setAlpha(alpha)
                pen.setColor(color)

                if hasattr(item, "label"):
                    item.label.setColor(color)

            style = settings.get(Updater.STYLE_LABEL)
            if style is not None:
                pen.setStyle(Updater.LINE_STYLES[style])

            width = settings.get(Updater.WIDTH_LABEL)
            if width is not None:
                pen.setWidth(width)

            item.setPen(pen)


class CommonParameterSetter:
    """ Subclass to add 'setter' functionality to a plot. """
    LABELS_BOX = "Fonts"
    ANNOT_BOX = "Annotations"
    PLOT_BOX = "Figure"

    FONT_FAMILY_LABEL = "Font family"
    AXIS_TITLE_LABEL = "Axis title"
    AXIS_TICKS_LABEL = "Axis ticks"
    LEGEND_LABEL = "Legend"
    LABEL_LABEL = "Label"
    LINE_LAB_LABEL = "Line label"
    X_AXIS_LABEL = "x-axis title"
    Y_AXIS_LABEL = "y-axis title"
    TITLE_LABEL = "Title"
    LINE_LABEL = "Lines"

    FONT_FAMILY_SETTING = None  # set in __init__ because it requires a running QApplication
    FONT_SETTING = None  # set in __init__ because it requires a running QApplication
    LINE_SETTING: SettingsType = {
        Updater.WIDTH_LABEL: (range(1, 15), 1),
        Updater.ALPHA_LABEL: (range(0, 255, 5), 255),
        Updater.STYLE_LABEL: (list(Updater.LINE_STYLES), Updater.DEFAULT_LINE_STYLE),
        Updater.ANTIALIAS_LABEL: (None, False),
    }

    def __init__(self):
        def update_font_family(**settings):
            # false positive, pylint: disable=unsubscriptable-object
            for label in self.initial_settings[self.LABELS_BOX]:
                if label != self.FONT_FAMILY_LABEL:
                    setter = self._setters[self.LABELS_BOX][label]
                    setter(**settings)

        def update_title(**settings):
            Updater.update_plot_title_font(self.title_item, **settings)

        def update_label(**settings):
            self.label_font = Updater.change_font(self.label_font, settings)
            Updater.update_label_font(self.labels, self.label_font)

        def update_axes_titles(**settings):
            Updater.update_axes_titles_font(self.axis_items, **settings)

        def update_axes_ticks(**settings):
            Updater.update_axes_ticks_font(self.axis_items, **settings)

        def update_legend(**settings):
            self.legend_settings.update(**settings)
            Updater.update_legend_font(self.legend_items, **settings)

        def update_title_text(**settings):
            Updater.update_plot_title_text(
                self.title_item, settings[self.TITLE_LABEL])

        def update_axis(axis, **settings):
            Updater.update_axis_title_text(
                self.getAxis(axis), settings[self.TITLE_LABEL])

        self.FONT_FAMILY_SETTING: SettingsType = {  # pylint: disable=invalid-name
            Updater.FONT_FAMILY_LABEL: (available_font_families(), default_font_family()),
        }

        self.FONT_SETTING: SettingsType = {  # pylint: disable=invalid-name
            Updater.SIZE_LABEL: (range(4, 50), QFont().pointSize()),
            Updater.IS_ITALIC_LABEL: (None, False)
        }

        self.label_font = QFont()
        self.legend_settings = {}

        self._setters = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: update_font_family,
                self.TITLE_LABEL: update_title,
                self.LABEL_LABEL: update_label,
                self.AXIS_TITLE_LABEL: update_axes_titles,
                self.AXIS_TICKS_LABEL: update_axes_ticks,
                self.LEGEND_LABEL: update_legend,
            },
            self.ANNOT_BOX: {
                self.TITLE_LABEL: update_title_text,
                self.X_AXIS_LABEL: lambda **kw: update_axis("bottom", **kw),
                self.Y_AXIS_LABEL: lambda **kw: update_axis("left", **kw),
            }
        }

        self.initial_settings: Dict[str, Dict[str, SettingsType]] = NotImplemented

        self.update_setters()
        self._check_setters()

    def update_setters(self):
        pass

    def _check_setters(self):
        # false positive, pylint: disable=not-an-iterable
        assert all(key in self._setters for key in self.initial_settings)
        for k, inner in self.initial_settings.items():
            assert all(key in self._setters[k] for key in inner)

    def set_parameter(self, key: KeyType, value: ValueType):
        self._setters[key[0]][key[1]](**{key[2]: value})
