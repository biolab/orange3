
from orangewidget.utils.filedialogs import (
    open_filename_dialog_save, open_filename_dialog,
    RecentPath, RecentPathsWidgetMixin, RecentPathsWComboMixin,
)
# imported for backcompatibility
from orangewidget.utils.filedialogs import (  # pylint: disable=unused-import
    fix_extension, format_filter, get_file_name, Compression
)

from Orange.data.io import FileFormat
from Orange.util import deprecated

__all__ = [
    "open_filename_dialog_save", "open_filename_dialog",
    "RecentPath", "RecentPathsWidgetMixin", "RecentPathsWComboMixin",
]


@deprecated
def dialog_formats():
    """
    Return readable file types for QFileDialogs.
    """
    return ("All readable files ({});;".format(
        '*' + ' *'.join(FileFormat.readers.keys())) +
            ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                      for f in sorted(set(FileFormat.readers.values()),
                                      key=list(FileFormat.readers.values()).index)))
