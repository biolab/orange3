import os.path
from operator import attrgetter

from PyQt4 import QtGui, QtCore


# noinspection PyBroadException
def save_plot(data, file_formats, filename=""):
    formats = [(f.DESCRIPTION, f.EXTENSIONS)
               for f in sorted(set(file_formats.values()),
                               key=attrgetter("OWSAVE_PRIORITY"))]
    filters = ['{} (*{})'.format(desc, ' *'.join(exts))
               for desc, exts in formats]

    _LAST_DIR_KEY = "directories/last_graph_directory"
    _LAST_EXT_KEY = "directories/last_graph_extension"
    settings = QtCore.QSettings()
    start_dir = settings.value(_LAST_DIR_KEY, filename)
    if not start_dir or not os.path.exists(start_dir):
        start_dir = os.path.expanduser("~")
    last_ext = settings.value(_LAST_EXT_KEY, "")
    if last_ext not in filters:
        last_ext = filters[0]

    filename, filter = QtGui.QFileDialog.getSaveFileNameAndFilter(
        None, 'Save as ...', start_dir, ';;'.join(filters), last_ext)
    if not filename:
        return

    ext = os.path.splitext(filename)[1]
    exts = formats[filters.index(filter)][1]
    if ext not in exts:
        ext = exts[0]
        filename += ext
    try:
        file_formats[ext].write(filename, data)
    except:
        QtGui.QMessageBox.critical(
            None, "Error", "Error occurred while saving file")

    settings.setValue(_LAST_DIR_KEY, os.path.split(filename)[0])
    settings.setValue(_LAST_EXT_KEY, filter)
