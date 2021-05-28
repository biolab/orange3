import math
from typing import Optional, Tuple

from AnyQt.QtCore import QModelIndex, QSize, Qt
from AnyQt.QtWidgets import QStyle, QStyleOptionViewItem, QApplication

from orangewidget.utils.itemdelegates import DataDelegate
from orangewidget.utils.cache import LRUCache
from orangewidget.gui import OrangeUserRole


class FixedFormatNumericColumnDelegate(DataDelegate):
    """
    A numeric delegate displaying in a fixed format.

    Parameters
    ----------
    ndecimals: int
        The number of decimals in the display
    ndigits: int
        The max number of digits in the integer part. If the model returns
        `ColumnDataSpanRole` data for a column then the `ndigits` is derived
        from that.

        .. note:: This is only used for size hinting.

    """
    ColumnDataSpanRole = next(OrangeUserRole)

    def __init__(self, *args, ndecimals=3, ndigits=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndecimals = ndecimals
        self.ndigits = ndigits
        self.__sh_cache = LRUCache(maxlen=200)
        self.__style = None

    def displayText(self, value, locale) -> str:
        if isinstance(value, self.RealTypes):
            return locale.toString(float(value), 'f', self.ndecimals)
        return super().displayText(value, locale)  # pragma: no cover

    def spanData(self, index: QModelIndex) -> Optional[Tuple[float, float]]:
        """
        Return the min, max numeric data values in the column that `index`
        is in.
        """
        span = self.cachedData(index, self.ColumnDataSpanRole)
        try:
            min_, max_ = span
        except (ValueError, TypeError):  # pragma: no cover
            return None
        if isinstance(min_, self.NumberTypes) and isinstance(max_, self.NumberTypes):
            return float(min_), float(max_)
        else:
            return None  # pragma: no cover

    @staticmethod
    def template(value: float, ndecimals=3) -> str:
        sign = math.copysign(1., value)
        # number of digits (integer part)
        ndigits = int(math.ceil(math.log10(abs(value) + 1)))
        template = "X" * ndigits + "." + "X" * ndecimals
        if sign == -1.:
            template = "-" + template
        return template

    def sizeHint(
            self, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QSize:
        widget = option.widget
        template = self.template(-10 ** self.ndigits, self.ndecimals)
        span = self.spanData(index)
        if span is not None:
            vmin, vmax = span
            t1 = self.template(vmin, self.ndecimals)
            t2 = self.template(vmax, self.ndecimals)
            template = max((t1, t2), key=len)
        style = widget.style() if widget is not None else QApplication.style()
        # Keep ref to style wrapper. This is ugly, wrong but the wrapping of
        # C++ QStyle instance takes ~5% unless the wrapper already exists.
        self.__style = style
        opt = QStyleOptionViewItem(option)
        opt.features |= QStyleOptionViewItem.HasDisplay
        sh = QSize()
        key = option.font.key(), template
        if key not in self.__sh_cache:
            for d in map(str, range(10)):
                opt.text = template.replace("X", d)
                sh_ = style.sizeFromContents(
                    QStyle.CT_ItemViewItem, opt, QSize(), widget)
                sh = sh.expandedTo(sh_)
            self.__sh_cache[key] = sh
        else:
            sh = self.__sh_cache[key]
        return QSize(sh)


class TableDataDelegate(DataDelegate):
    """
    A DataDelegate initialized to be used with
    :class:`Orange.widgets.utils.itemmodels.TableModel`
    """
    #: Roles supplied by TableModel we want DataDelegate to use.
    DefaultRoles = (Qt.DisplayRole, Qt.TextAlignmentRole, Qt.BackgroundRole)
