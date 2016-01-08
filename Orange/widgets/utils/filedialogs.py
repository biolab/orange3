import os

from PyQt4.QtGui import QMessageBox, QFileDialog


def fix_extension(ext, format, suggested_ext, suggested_format):
    dlg = QMessageBox(
        QMessageBox.Warning,
        "Mismatching extension",
        "Extension '{}' does not match the chosen file format, {}.\n\n"
        "Would you like to fix this?".format(ext, format))
    role = QMessageBox.AcceptRole
    change_ext = \
        suggested_ext and \
        dlg.addButton("Change extension to " + suggested_ext, role)
    change_format =\
        suggested_format and \
        dlg.addButton("Save as " + suggested_format, role)
    cancel = dlg.addButton("Back", role)
    dlg.setEscapeButton(cancel)
    dlg.exec()
    if dlg.clickedButton() == cancel:
        return fix_extension.CANCEL
    elif dlg.clickedButton() == change_ext:
        return fix_extension.CHANGE_EXT
    elif dlg.clickedButton() == change_format:
        return fix_extension.CHANGE_FORMAT

fix_extension.CHANGE_EXT = 0
fix_extension.CHANGE_FORMAT = 1
fix_extension.CANCEL = 2


def format_filter(writer):
    return '{} (*{})'.format(writer.DESCRIPTION, ' *'.join(writer.EXTENSIONS))


def get_file_name(start_dir, start_filter, file_formats):
    """
    Get filename for the given possible file formats

    The function uses the standard save file dialog with filters from the
    given file formats. Extension is added automatically, if missing. If the
    user enters file extension that does not match the file format, (s)he is
    given a dialog to decide whether to fix the extension or the format.

    Function also returns the writer and filter to cover the case where the
    same extension appears in multiple filters. Although `file_format` is a
    dictionary that associates its extension with one writer, writers can
    still have other extensions that are allowed.

    Args:
        start_dir (str): initial directory, optionally including the filename
        start_filter (str): initial filter
        file_formats (list of Orange.data.io.FileFormat): file formats
    Returns:
        (filename, filter, writer), or `(None, None, None)` on cancel
    """
    writers = sorted(set(file_formats.values()), key=lambda w: w.PRIORITY)
    filters = [format_filter(w) for w in writers]
    if start_filter not in filters:
        start_filter = filters[0]

    while True:
        filename, filter = QFileDialog.getSaveFileNameAndFilter(
            None, 'Save as ...', start_dir, ';;'.join(filters), start_filter)
        if not filename:
            return None, None, None

        writer = writers[filters.index(filter)]
        base, ext = os.path.splitext(filename)
        if not ext:
            filename += writer.EXTENSIONS[0]
        elif ext not in writer.EXTENSIONS:
            format = writer.DESCRIPTION
            suggested_ext = writer.EXTENSIONS[0]
            suggested_format = \
                ext in file_formats and file_formats[ext].DESCRIPTION
            res = fix_extension(ext, format, suggested_ext, suggested_format)
            if res == fix_extension.CANCEL:
                continue
            if res == fix_extension.CHANGE_EXT:
                filename = base + suggested_ext
            elif res == fix_extension.CHANGE_FORMAT:
                writer = file_formats[ext]
                filter = format_filter(writer)
        return filename, writer, filter
