import os.path

from PyQt4 import QtGui, QtCore


# noinspection PyBroadException
def save_plot(data, file_formats, filename=""):
    writers = sorted(set(file_formats.values()), key=lambda w: w.PRIORITY)
    filters = ['{} (*{})'.format(w.DESCRIPTION, ' *'.join(w.EXTENSIONS))
               for w in writers]

    _LAST_DIR_KEY = "directories/last_graph_directory"
    _LAST_FMT_KEY = "directories/last_graph_format"
    settings = QtCore.QSettings()
    start_dir = settings.value(_LAST_DIR_KEY, filename)
    if not start_dir or \
            (not os.path.exists(start_dir) and
             not os.path.exists(os.path.split(start_dir)[0])):
        start_dir = os.path.expanduser("~")
    last_ext = settings.value(_LAST_FMT_KEY, "")
    if last_ext not in filters:
        last_ext = filters[0]

    filename, filter = QtGui.QFileDialog.getSaveFileNameAndFilter(
        None, 'Save as ...', start_dir, ';;'.join(filters), last_ext)
    if not filename:
        return

    writer = writers[filters.index(filter)]
    if not os.path.splitext(filename)[1] and writer.EXTENSIONS:
        filename += writer.EXTENSIONS[0]
    try:
        writer.write(filename, data)
    except:
        QtGui.QMessageBox.critical(
            None, "Error", "Error occurred while saving file\n" + filename)

    settings.setValue(_LAST_DIR_KEY, os.path.split(filename)[0])
    settings.setValue(_LAST_FMT_KEY, filter)
