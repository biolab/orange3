import math

from AnyQt.QtCore import (
    Qt, QLineF, QModelIndex, QPoint, QSize, QLocale, QEvent,
    QPersistentModelIndex, QRect
)
from AnyQt.QtGui import (
    QIcon, QColor, QBrush, QPainter, QPen, QFont, QFontMetrics
)
from AnyQt.QtWidgets import (
    QApplication, QStyle, QSizePolicy, QLabel, QTableWidgetItem, QItemDelegate,
    QStyledItemDelegate, QStyleOptionViewItem
)

import Orange
from Orange.widgets.gui.base import OrangeUserRole

__all__ = ["tableItem",
           "TableValueRole", "TableClassValueRole", "TableDistribution",
           "TableVariable", "BarRatioRole", "BarBrushRole", "SortOrderRole",
           "LinkRole",
           "TableBarItem",
           "BarItemDelegate", "IndicatorItemDelegate", "LinkStyledItemDelegate",
           "ColoredBarItemDelegate", "HorizontalGridDelegate",
           "VerticalItemDelegate"]


# TODO: If nobody is using it, I'd prefer removing it
class tableItem(QTableWidgetItem):
    def __init__(self, table, x, y, text, editType=None, backColor=None,
                 icon=None, type=QTableWidgetItem.Type):
        super().__init__(type)
        if icon:
            self.setIcon(QIcon(icon))
        if editType is not None:
            self.setFlags(editType)
        else:
            self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable |
                          Qt.ItemIsSelectable)
        if backColor is not None:
            self.setBackground(QBrush(backColor))
        # we add it this way so that text can also be int and sorting will be
        # done properly (as integers and not as text)
        self.setData(Qt.DisplayRole, text)
        table.setItem(x, y, self)


TableValueRole = next(OrangeUserRole)  # Role to retrieve orange.Value
TableClassValueRole = next(OrangeUserRole)  # Retrieve class value for the row
TableDistribution = next(OrangeUserRole)  # Retrieve distribution of the column
TableVariable = next(OrangeUserRole)  # Role to retrieve the column's variable

BarRatioRole = next(OrangeUserRole)  # Ratio for drawing distribution bars
BarBrushRole = next(OrangeUserRole)  # Brush for distribution bar

SortOrderRole = next(OrangeUserRole)  # Used for sorting


class TableBarItem(QItemDelegate):
    BarRole = next(OrangeUserRole)
    BarColorRole = next(OrangeUserRole)

    def __init__(self, parent=None, color=QColor(255, 170, 127),
                 color_schema=None):
        """
        :param QObject parent: Parent object.
        :param QColor color: Default color of the distribution bar.
        :param color_schema:
            If not None it must be an instance of
            :class:`OWColorPalette.ColorPaletteGenerator` (note: this
            parameter, if set, overrides the ``color``)
        :type color_schema: :class:`OWColorPalette.ColorPaletteGenerator`
        """
        super().__init__(parent)
        self.color = color
        self.color_schema = color_schema

    def paint(self, painter, option, index):
        painter.save()
        self.drawBackground(painter, option, index)
        ratio = index.data(TableBarItem.BarRole)
        if isinstance(ratio, float):
            if math.isnan(ratio):
                ratio = None

        color = None
        if ratio is not None:
            if self.color_schema is not None:
                class_ = index.data(TableClassValueRole)
                if isinstance(class_, Orange.data.Value) and \
                        class_.variable.is_discrete and \
                        not math.isnan(class_):
                    color = self.color_schema[int(class_)]
            else:
                color = index.data(self.BarColorRole)
        if color is None:
            color = self.color
        rect = option.rect
        if ratio is not None:
            pw = 5
            hmargin = 3 + pw / 2  # + half pen width for the round line cap
            vmargin = 1
            textoffset = pw + vmargin * 2
            baseline = rect.bottom() - textoffset / 2
            width = (rect.width() - 2 * hmargin) * ratio
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QBrush(color), pw, Qt.SolidLine, Qt.RoundCap))
            line = QLineF(
                rect.left() + hmargin, baseline,
                rect.left() + hmargin + width, baseline)
            painter.drawLine(line)
            painter.restore()
            text_rect = rect.adjusted(0, 0, 0, -textoffset)
        else:
            text_rect = rect
        text = str(index.data(Qt.DisplayRole))
        self.drawDisplay(painter, option, text_rect, text)
        painter.restore()


class BarItemDelegate(QStyledItemDelegate):
    def __init__(self, parent, brush=QBrush(QColor(255, 170, 127)),
                 scale=(0.0, 1.0)):
        super().__init__(parent)
        self.brush = brush
        self.scale = scale

    def paint(self, painter, option, index):
        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter,
            option.widget)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter,
            option.widget)

        rect = option.rect
        val = index.data(Qt.DisplayRole)
        if isinstance(val, float):
            minv, maxv = self.scale
            val = (val - minv) / (maxv - minv)
            painter.save()
            if option.state & QStyle.State_Selected:
                painter.setOpacity(0.75)
            painter.setBrush(self.brush)
            painter.drawRect(
                rect.adjusted(1, 1, - rect.width() * (1.0 - val) - 2, -2))
            painter.restore()


class IndicatorItemDelegate(QStyledItemDelegate):
    IndicatorRole = next(OrangeUserRole)

    def __init__(self, parent, role=IndicatorRole, indicatorSize=2):
        super().__init__(parent)
        self.role = role
        self.indicatorSize = indicatorSize

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        rect = option.rect
        indicator = index.data(self.role)

        if indicator:
            painter.save()
            painter.setRenderHints(QPainter.Antialiasing)
            painter.setBrush(QBrush(Qt.black))
            painter.drawEllipse(rect.center(),
                                self.indicatorSize, self.indicatorSize)
            painter.restore()


class LinkStyledItemDelegate(QStyledItemDelegate):
    LinkRole = next(OrangeUserRole)

    def __init__(self, parent):
        super().__init__(parent)
        self.mousePressState = QModelIndex(), QPoint()
        parent.entered.connect(self.onEntered)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(size.height(), 20))

    def linkRect(self, option, index):
        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        text = self.displayText(index.data(Qt.DisplayRole), QLocale.system())
        self.initStyleOption(option, index)
        textRect = style.subElementRect(
            QStyle.SE_ItemViewItemText, option, option.widget)

        if not textRect.isValid():
            textRect = option.rect
        margin = style.pixelMetric(
            QStyle.PM_FocusFrameHMargin, option, option.widget) + 1
        textRect = textRect.adjusted(margin, 0, -margin, 0)
        font = index.data(Qt.FontRole)
        if not isinstance(font, QFont):
            font = option.font

        metrics = QFontMetrics(font)
        elideText = metrics.elidedText(text, option.textElideMode,
                                       textRect.width())
        return metrics.boundingRect(textRect, option.displayAlignment,
                                    elideText)

    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonPress and \
                self.linkRect(option, index).contains(event.pos()):
            self.mousePressState = (QPersistentModelIndex(index),
                                    QPoint(event.pos()))

        elif event.type() == QEvent.MouseButtonRelease:
            link = index.data(LinkRole)
            if not isinstance(link, str):
                link = None

            pressedIndex, pressPos = self.mousePressState
            if pressedIndex == index and \
                    (pressPos - event.pos()).manhattanLength() < 5 and \
                    link is not None:
                import webbrowser
                webbrowser.open(link)
            self.mousePressState = QModelIndex(), event.pos()

        elif event.type() == QEvent.MouseMove:
            link = index.data(LinkRole)
            if not isinstance(link, str):
                link = None

            if link is not None and \
                    self.linkRect(option, index).contains(event.pos()):
                self.parent().viewport().setCursor(Qt.PointingHandCursor)
            else:
                self.parent().viewport().setCursor(Qt.ArrowCursor)

        return super().editorEvent(event, model, option, index)

    def onEntered(self, index):
        link = index.data(LinkRole)
        if not isinstance(link, str):
            link = None
        if link is None:
            self.parent().viewport().setCursor(Qt.ArrowCursor)

    def paint(self, painter, option, index):
        link = index.data(LinkRole)
        if not isinstance(link, str):
            link = None

        if link is not None:
            if option.widget is not None:
                style = option.widget.style()
            else:
                style = QApplication.style()
            style.drawPrimitive(
                QStyle.PE_PanelItemViewRow, option, painter,
                option.widget)
            style.drawPrimitive(
                QStyle.PE_PanelItemViewItem, option, painter,
                option.widget)

            text = self.displayText(index.data(Qt.DisplayRole),
                                    QLocale.system())
            textRect = style.subElementRect(
                QStyle.SE_ItemViewItemText, option, option.widget)
            if not textRect.isValid():
                textRect = option.rect
            margin = style.pixelMetric(
                QStyle.PM_FocusFrameHMargin, option, option.widget) + 1
            textRect = textRect.adjusted(margin, 0, -margin, 0)
            elideText = QFontMetrics(option.font).elidedText(
                text, option.textElideMode, textRect.width())
            painter.save()
            font = index.data(Qt.FontRole)
            if not isinstance(font, QFont):
                font = option.font
            painter.setFont(font)
            if option.state & QStyle.State_Selected:
                color = option.palette.highlightedText().color()
            else:
                color = option.palette.link().color()
            painter.setPen(QPen(color))
            painter.drawText(textRect, option.displayAlignment, elideText)
            painter.restore()
        else:
            super().paint(painter, option, index)


LinkRole = LinkStyledItemDelegate.LinkRole


class ColoredBarItemDelegate(QStyledItemDelegate):
    """ Item delegate that can also draws a distribution bar
    """
    def __init__(self, parent=None, decimals=3, color=Qt.red):
        super().__init__(parent)
        self.decimals = decimals
        self.float_fmt = "%%.%if" % decimals
        self.color = QColor(color)

    def displayText(self, value, locale=QLocale()):
        if value is None or isinstance(value, float) and math.isnan(value):
            return "NA"
        if isinstance(value, float):
            return self.float_fmt % value
        return str(value)

    def sizeHint(self, option, index):
        font = self.get_font(option, index)
        metrics = QFontMetrics(font)
        height = metrics.lineSpacing() + 8  # 4 pixel margin
        width = metrics.width(
            self.displayText(index.data(Qt.DisplayRole), QLocale())) + 8
        return QSize(width, height)

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        text = self.displayText(index.data(Qt.DisplayRole))
        ratio, have_ratio = self.get_bar_ratio(option, index)

        rect = option.rect
        if have_ratio:
            # The text is raised 3 pixels above the bar.
            # TODO: Style dependent margins?
            text_rect = rect.adjusted(4, 1, -4, -4)
        else:
            text_rect = rect.adjusted(4, 4, -4, -4)

        painter.save()
        font = self.get_font(option, index)
        painter.setFont(font)

        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter,
            option.widget)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter,
            option.widget)

        # TODO: Check ForegroundRole.
        if option.state & QStyle.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
        painter.setPen(QPen(color))

        align = self.get_text_align(option, index)

        metrics = QFontMetrics(font)
        elide_text = metrics.elidedText(
            text, option.textElideMode, text_rect.width())
        painter.drawText(text_rect, align, elide_text)

        painter.setRenderHint(QPainter.Antialiasing, True)
        if have_ratio:
            brush = self.get_bar_brush(option, index)

            painter.setBrush(brush)
            painter.setPen(QPen(brush, 1))
            bar_rect = QRect(text_rect)
            bar_rect.setTop(bar_rect.bottom() - 1)
            bar_rect.setBottom(bar_rect.bottom() + 1)
            w = text_rect.width()
            bar_rect.setWidth(max(0, min(w * ratio, w)))
            painter.drawRoundedRect(bar_rect, 2, 2)
        painter.restore()

    def get_font(self, option, index):
        font = index.data(Qt.FontRole)
        if not isinstance(font, QFont):
            font = option.font
        return font

    def get_text_align(self, _, index):
        align = index.data(Qt.TextAlignmentRole)
        if not isinstance(align, int):
            align = Qt.AlignLeft | Qt.AlignVCenter

        return align

    def get_bar_ratio(self, _, index):
        ratio = index.data(BarRatioRole)
        return ratio, isinstance(ratio, float)

    def get_bar_brush(self, _, index):
        bar_brush = index.data(BarBrushRole)
        if not isinstance(bar_brush, (QColor, QBrush)):
            bar_brush = self.color
        return QBrush(bar_brush)


class HorizontalGridDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        painter.setPen(QColor(212, 212, 212))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        painter.restore()
        QStyledItemDelegate.paint(self, painter, option, index)


class VerticalItemDelegate(QStyledItemDelegate):
    # Extra text top/bottom margin.
    Margin = 6

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        return QSize(sh.height() + self.Margin * 2, sh.width())

    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)

        if not option.text:
            return

        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()
        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter,
            option.widget)
        cell_rect = option.rect
        itemrect = QRect(0, 0, cell_rect.height(), cell_rect.width())
        opt = QStyleOptionViewItem(option)
        opt.rect = itemrect
        textrect = style.subElementRect(
            QStyle.SE_ItemViewItemText, opt, opt.widget)

        painter.save()
        painter.setFont(option.font)

        if option.displayAlignment & (Qt.AlignTop | Qt.AlignBottom):
            brect = painter.boundingRect(
                textrect, option.displayAlignment, option.text)
            diff = textrect.height() - brect.height()
            offset = max(min(diff / 2, self.Margin), 0)
            if option.displayAlignment & Qt.AlignBottom:
                offset = -offset

            textrect.translate(0, offset)

        painter.translate(option.rect.x(), option.rect.bottom())
        painter.rotate(-90)
        painter.drawText(textrect, option.displayAlignment, option.text)
        painter.restore()
