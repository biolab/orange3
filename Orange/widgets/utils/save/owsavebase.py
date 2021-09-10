import os.path
import sys
import re

from AnyQt.QtWidgets import QFileDialog, QGridLayout, QMessageBox

from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting


_userhome = os.path.expanduser(f"~{os.sep}")


class OWSaveBase(widget.OWWidget, openclass=True):
    """
    Base class for Save widgets

    A derived class must provide, at minimum:

    - class `Inputs` and the corresponding handler that:

      - saves the input to an attribute `data`, and
      - calls `self.on_new_input`.

    - a class attribute `filters` with a list of filters or a dictionary whose
      keys are filters OR a class method `get_filters` that returns such a
      list or dictionary
    - method `do_save` that saves `self.data` into `self.filename`

    Alternatively, instead of defining `do_save` a derived class can make
    `filters` a dictionary whose keys are classes that define a method `write`
    (like e.g. `TabReader`). Method `do_save` defined in the base class calls
    the writer corresponding to the currently chosen filter.

    A minimum example of derived class is
    `Orange.widgets.model.owsavemodel.OWSaveModel`.
    A more advanced widget that overrides a lot of base class behaviour is
    `Orange.widgets.data.owsave.OWSave`.
    """

    class Information(widget.OWWidget.Information):
        empty_input = widget.Msg("Empty input; nothing was saved.")

    class Error(widget.OWWidget.Error):
        no_file_name = widget.Msg("File name is not set.")
        unsupported_format = widget.Msg("File format is unsupported.\n{}")
        general_error = widget.Msg("{}")

    want_main_area = False
    resizing_enabled = False

    filter = Setting("")  # Default is provided in __init__

    # If the path is in the same directory as the workflow file or its
    # subdirectory, it is stored as a relative path, otherwise as absolute.
    # For the sake of security, we do not store relative paths from other
    # directories, like home or cwd. After loading the widget from a schema,
    # auto_save is set to off, unless the stored_path is relative (to the
    # workflow).
    stored_path = Setting("")
    stored_name = Setting("", schema_only=True)  # File name, without path
    auto_save = Setting(False)

    filters = []

    def __init__(self, start_row=0):
        """
        Set up the gui.

        The gui consists of a checkbox for auto save and two buttons put on a
        grid layout. Derived widgets that want to place controls above the auto
        save widget can set the `start_row` argument to the first free row,
        and this constructor will start filling the grid there.

        Args:
            start_row (int): the row at which to start filling the gui
        """
        super().__init__()
        self.data = None
        self._absolute_path = self._abs_path_from_setting()

        # This cannot be done outside because `filters` is defined by subclass
        if not self.filter:
            self.filter = self.default_filter()

        self.grid = grid = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=grid, box=True)
        grid.addWidget(
            gui.checkBox(
                None, self, "auto_save", "Autosave when receiving new data",
                callback=self.update_messages),
            start_row, 0, 1, 2)
        self.bt_save = gui.button(
            self.buttonsArea, self,
            label=f"Save as {self.stored_name}" if self.stored_name else "Save",
            callback=self.save_file)
        gui.button(self.buttonsArea, self, "Save as ...", callback=self.save_file_as)

        self.adjustSize()
        self.update_messages()

    def default_filter(self):
        """Returns the first filter in the list"""
        return next(iter(self.get_filters()))

    @property
    def last_dir(self):
        # Not the best name, but kept for compatibility
        return self._absolute_path

    @last_dir.setter
    def last_dir(self, absolute_path):
        """Store _absolute_path and update relative path (stored_path)"""
        self._absolute_path = absolute_path

        workflow_dir = self.workflowEnv().get("basedir", None)
        if workflow_dir and absolute_path.startswith(workflow_dir.rstrip("/")):
            self.stored_path = os.path.relpath(absolute_path, workflow_dir)
        else:
            self.stored_path = absolute_path

    def _abs_path_from_setting(self):
        """
        Compute absolute path from `stored_path` from settings.

        Absolute stored path is used only if it exists.
        Auto save is disabled unless stored_path is relative.
        """
        workflow_dir = self.workflowEnv().get("basedir")
        if os.path.isabs(self.stored_path):
            if os.path.exists(self.stored_path):
                self.auto_save = False
                return self.stored_path
        elif workflow_dir is not None:
            return os.path.normpath(
                os.path.join(workflow_dir, self.stored_path))

        self.stored_path = workflow_dir or _userhome
        self.auto_save = False
        return self.stored_path

    @property
    def filename(self):
        if self.stored_name:
            return os.path.join(self._absolute_path, self.stored_name)
        else:
            return ""

    @filename.setter
    def filename(self, value):
        self.last_dir, self.stored_name = os.path.split(value)

    # pylint: disable=unused-argument
    def workflowEnvChanged(self, key, value, oldvalue):
        # Trigger refresh of relative path, e.g. when saving the scheme
        if key == "basedir":
            self.last_dir = self._absolute_path

    @classmethod
    def get_filters(cls):
        return cls.filters

    @property
    def writer(self):
        """
        Return the active writer or None if there is no writer for this filter

        The base class uses this property only in `do_save` to find the writer
        corresponding to the filter. Derived classes (e.g. OWSave) may also use
        it elsewhere.

        Filter may not exist if it comes from settings saved in Orange with
        some add-ons that are not (or no longer) present, or if support for
        some extension was dropped, like the old Excel format.
        """
        filters = self.get_filters()
        if self.filter not in filters:
            return None
        return filters[self.filter]

    def on_new_input(self):
        """
        This method must be called from input signal handler.

        - It clears errors, warnings and information and calls
          `self.update_messages` to set the as needed.
        - It also calls `update_status` the can be overriden in derived
          methods to set the status (e.g. the number of input rows)
        - Calls `self.save_file` if `self.auto_save` is enabled and
          `self.filename` is provided.
        """
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self.update_messages()
        self.update_status()
        if self.auto_save and self.filename:
            self.save_file()

    def save_file_as(self):
        """
        Ask the user for the filename and try saving the file
        """
        filename, selected_filter = self.get_save_filename()
        if not filename:
            return
        self.filename = filename
        self.filter = selected_filter
        self.Error.unsupported_format.clear()
        self.bt_save.setText(f"Save as {self.stored_name}")
        self.update_messages()
        self._try_save()

    def save_file(self):
        """
        If file name is provided, try saving, else call save_file_as
        """
        if not self.filename:
            self.save_file_as()
        else:
            self._try_save()

    def _try_save(self):
        """
        Private method that calls do_save within try-except that catches and
        shows IOError. Do nothing if not data or no file name.
        """
        self.Error.general_error.clear()
        if self.data is None or not self.filename:
            return
        try:
            self.do_save()
        except IOError as err_value:
            self.Error.general_error(str(err_value))

    def do_save(self):
        """
        Do the saving.

        Default implementation calls the write method of the writer
        corresponding to the current filter. This requires that get_filters()
        returns is a dictionary whose keys are classes.

        Derived classes may simplify this by providing a list of filters and
        override do_save. This is particularly handy if the widget supports only
        a single format.
        """
        # This method is separated out because it will usually be overriden
        if self.writer is None:
            self.Error.unsupported_format(self.filter)
            return
        self.writer.write(self.filename, self.data)

    def update_messages(self):
        """
        Update errors, warnings and information.

        Default method sets no_file_name if auto_save is enabled but file name
        is not provided; and empty_input if file name is given but there is no
        data.

        Derived classes that define further messages will typically set them in
        this method.
        """
        self.Error.no_file_name(shown=not self.filename and self.auto_save)
        self.Information.empty_input(shown=self.filename and self.data is None)

    def update_status(self):
        """
        Update the input/output indicator. Default method does nothing.
        """

    def initial_start_dir(self):
        """
        Provide initial start directory

        Return either the current file's path, the last directory or home.
        """
        if self.filename and os.path.exists(os.path.split(self.filename)[0]):
            return self.filename
        else:
            return self.last_dir or _userhome

    @staticmethod
    def suggested_name():
        """
        Suggest the name for the output file or return an empty string.
        """
        return ""

    @classmethod
    def _replace_extension(cls, filename, extension):
        """
        Remove all extensions that appear in any filter.

        Double extensions are broken in different weird ways across all systems,
        including omitting some, like turning iris.tab.gz to iris.gz. This
        function removes anything that can appear anywhere.
        """
        known_extensions = set()
        for filt in cls.get_filters():
            known_extensions |= set(cls._extension_from_filter(filt).split("."))
        if "" in known_extensions:
            known_extensions.remove("")
        while True:
            base, ext = os.path.splitext(filename)
            if ext[1:] not in known_extensions:
                break
            filename = base
        return filename + extension

    @staticmethod
    def _extension_from_filter(selected_filter):
        return re.search(r".*\(\*?(\..*)\)$", selected_filter).group(1)

    def valid_filters(self):
        return self.get_filters()

    def default_valid_filter(self):
        return self.filter

    @classmethod
    def migrate_settings(cls, settings, version):
        # We cannot use versions because they are overriden in derived classes
        if "last_dir" in settings:
            settings["stored_path"] = settings.pop("last_dir")
        if "filename" in settings:
            settings["stored_name"] = os.path.split(
                settings.pop("filename") or "")[1]

    # As of Qt 5.9, QFileDialog.setDefaultSuffix does not support double
    # suffixes, not even in non-native dialogs. We handle each OS separately.
    if sys.platform in ("darwin", "win32"):
        # macOS and Windows native dialogs do not correctly handle double
        # extensions. We thus don't pass any suffixes to the dialog and add
        # the correct suffix after closing the dialog and only then check
        # if the file exists and ask whether to override.
        # It is a bit confusing that the user does not see the final name in the
        # dialog, but I see no better solution.
        def get_save_filename(self):  # pragma: no cover
            if sys.platform == "darwin":
                def remove_star(filt):
                    return filt.replace(" (*.", " (.")
            else:
                def remove_star(filt):
                    return filt

            no_ext_filters = {remove_star(f): f for f in self.valid_filters()}
            filename = self.initial_start_dir()
            while True:
                dlg = QFileDialog(
                    None, "Save File", filename, ";;".join(no_ext_filters))
                dlg.setAcceptMode(dlg.AcceptSave)
                dlg.selectNameFilter(remove_star(self.default_valid_filter()))
                dlg.setOption(QFileDialog.DontConfirmOverwrite)
                if dlg.exec() == QFileDialog.Rejected:
                    return "", ""
                filename = dlg.selectedFiles()[0]
                selected_filter = no_ext_filters[dlg.selectedNameFilter()]
                filename = self._replace_extension(
                    filename, self._extension_from_filter(selected_filter))
                if not os.path.exists(filename) or QMessageBox.question(
                        self, "Overwrite file?",
                        f"File {os.path.split(filename)[1]} already exists.\n"
                        "Overwrite?") == QMessageBox.Yes:
                    return filename, selected_filter

    else:  # Linux and any unknown platforms
        # Qt does not use a native dialog on Linux, so we can connect to
        # filterSelected and to overload selectFile to change the extension
        # while the dialog is open.
        # For unknown platforms (which?), we also use the non-native dialog to
        # be sure we know what happens.
        class SaveFileDialog(QFileDialog):
            # pylint: disable=protected-access
            def __init__(self, save_cls, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.save_cls = save_cls
                self.suffix = ""
                self.setAcceptMode(QFileDialog.AcceptSave)
                self.setOption(QFileDialog.DontUseNativeDialog)
                self.filterSelected.connect(self.updateDefaultExtension)

            def selectNameFilter(self, selected_filter):
                super().selectNameFilter(selected_filter)
                self.updateDefaultExtension(selected_filter)

            def updateDefaultExtension(self, selected_filter):
                self.suffix = \
                    self.save_cls._extension_from_filter(selected_filter)
                files = self.selectedFiles()
                if files and not os.path.isdir(files[0]):
                    self.selectFile(files[0])

            def selectFile(self, filename):
                filename = \
                    self.save_cls._replace_extension(filename, self.suffix)
                super().selectFile(filename)

        def get_save_filename(self):
            dlg = self.SaveFileDialog(
                type(self),
                None, "Save File", self.initial_start_dir(),
                ";;".join(self.valid_filters()))
            dlg.selectNameFilter(self.default_valid_filter())
            if dlg.exec() == QFileDialog.Rejected:
                return "", ""
            else:
                return dlg.selectedFiles()[0], dlg.selectedNameFilter()
