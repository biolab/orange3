import os.path

from AnyQt.QtCore import QSettings

from Orange.widgets.utils import filedialogs


# noinspection PyBroadException
def save_plot(data, file_formats, filename=""):
    _LAST_DIR_KEY = "directories/last_graph_directory"
    _LAST_FILTER_KEY = "directories/last_graph_filter"
    settings = QSettings()
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
    writer.write(filename, data)
    settings.setValue(_LAST_DIR_KEY, os.path.split(filename)[0])
    settings.setValue(_LAST_FILTER_KEY, filter)


def main():  # pragma: no cover
    from AnyQt.QtWidgets import QApplication
    from Orange.widgets.widget import OWWidget
    app = QApplication([])
    save_plot(None, OWWidget.graph_writers)


if __name__ == "__main__":  # pragma: no cover
    main()
