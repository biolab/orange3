"""
Preview item model.
"""

import logging

from PyQt4.QtGui import (
    QStandardItemModel, QStandardItem, QStyledItemDelegate, QIcon
)

from PyQt4.QtCore import Qt, QTimer

from . import scanner

log = logging.getLogger(__name__)

# Preview Data Roles
####################

# Name of the item, (same as `Qt.DisplayRole`)
NameRole = Qt.DisplayRole

# Items description (items long description)
DescriptionRole = Qt.UserRole + 1

# Items url/path (where previewed resource is located).
PathRole = Qt.UserRole + 2

# Items preview SVG contents string
ThumbnailSVGRole = Qt.UserRole + 3


UNKNOWN_SVG = \
"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="161.8mm" height="100.0mm"
 xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
 version="1.2" baseProfile="tiny">
</svg>
"""


class PreviewModel(QStandardItemModel):
    """A model for preview items.
    """

    def __init__(self, parent=None, items=None):
        QStandardItemModel.__init__(self, parent)

        if items is not None:
            self.insertColumn(0, items)

    def delayedScanUpdate(self, delay=10):
        """Run a delayed preview item scan update.
        """
        def iter_update(items):
            for item in items:
                try:
                    scanner.scan_update(item)
                except Exception:
                    log.error("An unexpected error occurred while "
                              "scanning %r.", str(item.text()),
                              exc_info=True)
                    item.setEnabled(False)
                yield

        items = [self.item(i) for i in range(self.rowCount())]

        iter_scan = iter_update(items)

        def process_one():
            try:
                next(iter_scan)
                QTimer.singleShot(delay, process_one)
            except StopIteration:
                pass

        QTimer.singleShot(delay, process_one)


class PreviewItem(QStandardItem):
    """A preview item.
    """
    def __init__(self, name=None, description=None, thumbnail=None,
                 icon=None, path=None):
        QStandardItem.__init__(self)

        self.__name = ""

        if name is None:
            name = "Untitled"

        self.setName(name)

        if description is None:
            description = "No description."
        self.setDescription(description)

        if thumbnail is None:
            thumbnail = UNKNOWN_SVG
        self.setThumbnail(thumbnail)

        if icon is not None:
            self.setIcon(icon)

        if path is not None:
            self.setPath(path)

    def name(self):
        """Return the name (title) of the item (same as `text()`.
        """
        return self.__name

    def setName(self, value):
        """Set the item name. `value` if not empty will be used as
        the items DisplayRole otherwise an 'untitled' placeholder will
        be used.

        """
        self.__name = value

        if not value:
            self.setText("untitled")
        else:
            self.setText(value)

    def description(self):
        """Return the detailed description for the item.

        This is stored as `DescriptionRole`, if no data is set then
        return the string for `WhatsThisRole`.

        """
        desc = self.data(DescriptionRole)

        if desc is not None:
            return desc

        whatsthis = self.data(Qt.WhatsThisRole)
        return whatsthis

    def setDescription(self, description):
        self.setData(description, DescriptionRole)
        self.setWhatsThis(description)

    def thumbnail(self):
        """Return the thumbnail SVG string for the preview item.

        This is stored as `ThumbnailSVGRole`
        """
        thumb = self.data(ThumbnailSVGRole)
        if thumb is not None:
            return thumb

    def setThumbnail(self, thumbnail):
        """Set the thumbnail SVG contents as a string.

        When set it also overrides the icon role.

        """
        self.setData(thumbnail, ThumbnailSVGRole)

        # TODO: how to set the icon from the svg contents string
        # without writing it to disc
        import tempfile
        with tempfile.NamedTemporaryFile(
                prefix="thumbnail", suffix=".svg", delete=False
                ) as f:
            f.write(thumbnail.encode("utf-8"))
            f.flush()
            icon = QIcon(f.name)
            self.setIcon(icon)

    def path(self):
        """Return the path item data.
        """
        return self.data(PathRole)

    def setPath(self, path):
        """Set the path data of the item.

        .. note:: This also sets the Qt.StatusTipRole

        """
        self.setData(path, PathRole)
        self.setStatusTip(path)
        self.setToolTip(path)


class PreviewItemDelegate(QStyledItemDelegate):
    def __init__(self, ):
        QStyledItemDelegate.__init__(self,)
