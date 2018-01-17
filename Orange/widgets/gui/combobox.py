from AnyQt.QtWidgets import QComboBox
from AnyQt.QtCore import Qt, QSize

from Orange.widgets.utils import getdeepattr

from .base import miscellanea, log
from .boxes import widgetBox
from .labels import widgetLabel
from .callbacks import connect_control

__all__ = ["comboBox", "OrangeComboBox"]


class OrangeComboBox(QComboBox):
    """
    A QComboBox subclass extended to support bounded contents width hint.
    """
    def __init__(self, parent=None, maximumContentsLength=-1, **kwargs):
        # Forward-declared for sizeHint()
        self.__maximumContentsLength = maximumContentsLength
        super().__init__(parent, **kwargs)

    def setMaximumContentsLength(self, length):
        """
        Set the maximum contents length hint.

        The hint specifies the upper bound on the `sizeHint` and
        `minimumSizeHint` width specified in character length.
        Set to 0 or negative value to disable.

        .. note::
             This property does not affect the widget's `maximumSize`.
             The widget can still grow depending in it's sizePolicy.

        Parameters
        ----------
        lenght : int
            Maximum contents length hint.
        """
        if self.__maximumContentsLength != length:
            self.__maximumContentsLength = length
            self.updateGeometry()

    def maximumContentsLength(self):
        """
        Return the maximum contents length hint.
        """
        return self.__maximumContentsLength

    def sizeHint(self):
        # reimplemented
        sh = super().sizeHint()
        if self.__maximumContentsLength > 0:
            width = (self.fontMetrics().width("X") * self.__maximumContentsLength
                     + self.iconSize().width() + 4)
            sh = sh.boundedTo(QSize(width, sh.height()))
        return sh

    def minimumSizeHint(self):
        # reimplemented
        sh = super().minimumSizeHint()
        if self.__maximumContentsLength > 0:
            width = (self.fontMetrics().width("X") * self.__maximumContentsLength
                     + self.iconSize().width() + 4)
            sh = sh.boundedTo(QSize(width, sh.height()))
        return sh

    def setValueType(self, _):
        log.warning("OrangeComboBox.setValueType is obsolete and ignored")


def comboBox(widget, master, value, box=None, label=None, labelWidth=None,
             orientation=Qt.Vertical, items=(), callback=None,
             sendSelectedValue=None, emptyString=None, editable=False,
             contentsLength=None, maximumContentsLength=25,
             *, model=None,
             **misc):
    """
    Construct a combo box.

    The `value` attribute of the `master` contains the text or the
    index of the selected item.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param orientation: tells whether to put the label above or to the left
    :type orientation: `Qt.Horizontal` (default), `Qt.Vertical` or
        instance of `QLayout`
    :param label: a label that is inserted into the box
    :type label: str
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param callback: a function that is called when the value is changed
    :type callback: function
    :param items: items (optionally with data) that are put into the box
    :type items: tuple of str or tuples
    :param sendSelectedValue: decides whether the `value` contains the text
        of the selected item (`True`) or its index (`False`). If omitted
        (or `None`), the type will match the current value type, or index,
        if the current value is `None`.
    :type sendSelectedValue: bool or `None`
    :param emptyString: the string value in the combo box that gets stored as
        an empty string in `value`
    :type emptyString: str
    :param editable: a flag telling whether the combo is editable
    :type editable: bool
    :param int contentsLength: Contents character length to use as a
        fixed size hint. When not None, equivalent to::

            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(contentsLength)
    :param int maximumContentsLength: Specifies the upper bound on the
        `sizeHint` and `minimumSizeHint` width specified in character
        length (default: 25, use 0 to disable)
    :rtype: QComboBox
    """
    # Local import to avoid circular imports
    from Orange.widgets.utils.itemmodels import VariableListModel

    def update_combo(val):
        if val == "":  # the latter accomodates PyListModel
            val = None
        if val is None and None not in model:
            return  # e.g. attribute x in uninitialized scatter plot
        if val in model:
            combo.setCurrentIndex(model.indexOf(val))
            return
        elif isinstance(val, str):
            for i, item in enumerate(model):
                if val == str(item):
                    combo.setCurrentIndex(i)
                    return
        if value:
            raise ValueError("Combo box does not contain item %s" % repr(val))

    def update_attribute(val):
        def update_str():
            items = [combo.itemText(i) for i in range(combo.count())]
            try:
                index = items.index(val or emptyString)
            except ValueError:
                if val:
                    log.warning("Unable to set '%s' to '%s';"
                                "valid values are '%s'",
                                combo, val, ", ".join(items))
            else:
                combo.setCurrentIndex(index)

        def update_int():
            if val < combo.count():
                combo.setCurrentIndex(val)

        if isinstance(val, int):
            update_int()
        else:
            update_str()

    if box or label:
        hb = widgetBox(widget, box, orientation, addToLayout=False)
        if label is not None:
            widgetLabel(hb, label, labelWidth)
    else:
        hb = widget

    combo = OrangeComboBox(
        hb, maximumContentsLength=maximumContentsLength,
        editable=editable)

    if contentsLength is not None:
        combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(contentsLength)

    combo.box = hb
    for item in items:
        if isinstance(item, (tuple, list)):
            combo.addItem(*item)
        else:
            combo.addItem(str(item))

    if value:
        cindex = getdeepattr(master, value)
        if model is not None:
            combo.setModel(model)
        if isinstance(model, VariableListModel):
            update_combo(cindex)
            connect_control(
                master, value, combo,
                signal=combo.activated[int],
                update_control=update_combo,
                update_value=lambda index: setattr(master, value, model[index]),
                callback=callback)
        else:
            if isinstance(cindex, str):
                # If the current value is not a string, we leave it None for
                # backward compatibility.
                if sendSelectedValue is None:
                    sendSelectedValue = True
                if items and cindex in items:
                    cindex = items.index(cindex)
                else:
                    cindex = 0
            if cindex > combo.count() - 1:
                cindex = 0
            combo.setCurrentIndex(cindex)
            if sendSelectedValue:
                connect_control(
                    master, value, combo,
                    signal=combo.activated[str],
                    update_control=update_attribute,
                    update_value=lambda val: setattr(master, value, str(val) or emptyString),
                    callback=callback)
            else:
                connect_control(
                    master, value, combo,
                    signal=combo.activated[int],
                    update_control=update_attribute,
                    callback=callback)
    miscellanea(combo, hb, widget, **misc)
    combo.emptyString = emptyString
    return combo
