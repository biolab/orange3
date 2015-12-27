import os.path

from PyQt4 import QtGui, QtCore


CHANGE_EXT, CHANGE_FORMAT, KEEP = range(3)


def fix_extension(ext, format, suggested_ext, suggested_format):
    dlg = QtGui.QMessageBox(
        QtGui.QMessageBox.Warning,
        "Mismatching extension",
        "Extension '{}' does not match the chosen file format, {}.\n\n"
        "Would you like to fix this?".format(ext, format))
    role = QtGui.QMessageBox.AcceptRole
    keep_settings = dlg.addButton("Save as it is", role)
    change_ext = \
        suggested_ext and dlg.addButton("Use extension " + suggested_ext, role)
    change_format = \
        suggested_format and dlg.addButton("Save as " + suggested_format, role)
    dlg.exec()
    if dlg.clickedButton() == keep_settings:
        return KEEP
    elif dlg.clickedButton() == change_ext:
        return CHANGE_EXT
    elif dlg.clickedButton() == change_format:
        return CHANGE_FORMAT


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
    base, ext = os.path.splitext(filename)
    if not ext:
        filename += writer.EXTENSIONS[0]
    elif ext not in writer.EXTENSIONS:
        format = writer.DESCRIPTION
        suggested_ext = writer.EXTENSIONS[0]
        suggested_format = ext in file_formats and file_formats[ext].DESCRIPTION
        res = fix_extension(ext, format, suggested_ext, suggested_format)
        if res == CHANGE_EXT:
            filename = base + suggested_ext
        elif res == CHANGE_FORMAT:
            writer = file_formats[ext]

    try:
        writer.write(filename, data)
    except:
        QtGui.QMessageBox.critical(
            None, "Error", "Error occurred while saving file\n" + filename)

    settings.setValue(_LAST_DIR_KEY, os.path.split(filename)[0])
    settings.setValue(_LAST_FMT_KEY, filter)


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
