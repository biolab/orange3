import os.path

from PyQt4 import QtGui, QtCore

from Orange.widgets.utils import filedialogs


# noinspection PyBroadException
def save_plot(data, file_formats, filename=""):
    _LAST_DIR_KEY = "directories/last_graph_directory"
    _LAST_FILTER_KEY = "directories/last_graph_filter"
    settings = QtCore.QSettings()
    start_dir = settings.value(_LAST_DIR_KEY, filename)
    if not start_dir or \
            (not os.path.exists(start_dir) and
             not os.path.exists(os.path.split(start_dir)[0])):
        start_dir = os.path.expanduser("~")
    last_filter = settings.value(_LAST_FILTER_KEY, "")
    filename, writer, filter = \
        filedialogs.get_file_name(start_dir, last_filter, file_formats)
    if not filename:
        return
    try:
        writer.write(filename, data)
    except:
        QtGui.QMessageBox.critical(
            None, "Error", "Error occurred while saving file\n" + filename)
    else:
        settings.setValue(_LAST_DIR_KEY, os.path.split(filename)[0])
        settings.setValue(_LAST_FILTER_KEY, filter)


if __name__ == "__main__":
    from Orange.widgets.widget import OWWidget

    app = QtGui.QApplication([])

    save_plot(None, OWWidget.graph_writers)
    """
    ow = _ChangeExtension(".png", "Scalable Vector Graphics",
                          ".svg", "Portable Network Graphics")
    ow.exec()
    print(ow.clickedButton() == ow.change_ext)
"""
