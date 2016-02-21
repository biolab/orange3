import os

from PyQt4.QtGui import QMessageBox, QFileDialog, QFileIconProvider

from Orange.widgets.settings import Setting


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


class RecentPath:
    abspath = ''
    prefix = None   #: Option[str]  # BASEDIR | SAMPLE-DATASETS | ...
    relpath = ''  #: Option[str]  # path relative to `prefix`
    title = ''    #: Option[str]  # title of filename (e.g. from URL)

    def __init__(self, abspath, prefix, relpath, title=''):
        if os.name == "nt":
            # always use a cross-platform pathname component separator
            abspath = abspath.replace(os.path.sep, "/")
            if relpath is not None:
                relpath = relpath.replace(os.path.sep, "/")
        self.abspath = abspath
        self.prefix = prefix
        self.relpath = relpath
        self.title = title

    def __eq__(self, other):
        return (self.abspath == other.abspath or
                (self.prefix is not None and self.relpath is not None and
                 self.prefix == other.prefix and
                 self.relpath == other.relpath))

    @staticmethod
    def create(path, searchpaths):
        """
        Create a RecentPath item inferring a suitable prefix name and relpath.

        Parameters
        ----------
        path : str
            File system path.
        searchpaths : List[Tuple[str, str]]
            A sequence of (NAME, prefix) pairs. The sequence is searched
            for a item such that prefix/relpath == abspath. The NAME is
            recorded in the `prefix` and relpath in `relpath`.
            (note: the first matching prefixed path is chosen).

        """
        def isprefixed(prefix, path):
            """
            Is `path` contained within the directory `prefix`.

            >>> isprefixed("/usr/local/", "/usr/local/shared")
            True
            """
            normalize = lambda path: os.path.normcase(os.path.normpath(path))
            prefix, path = normalize(prefix), normalize(path)
            if not prefix.endswith(os.path.sep):
                prefix = prefix + os.path.sep
            return os.path.commonprefix([prefix, path]) == prefix

        abspath = os.path.normpath(os.path.abspath(path))
        for prefix, base in searchpaths:
            if isprefixed(base, abspath):
                relpath = os.path.relpath(abspath, base)
                return RecentPath(abspath, prefix, relpath)

        return RecentPath(abspath, None, None)

    def search(self, searchpaths):
        """
        Return a file system path, substituting the variable paths if required

        If the self.abspath names an existing path it is returned. Else if
        the `self.prefix` and `self.relpath` are not `None` then the
        `searchpaths` sequence is searched for the matching prefix and
        if found and the {PATH}/self.relpath exists it is returned.

        If all fails return None.

        Parameters
        ----------
        searchpaths : List[Tuple[str, str]]
            A sequence of (NAME, prefixpath) pairs.

        """
        if os.path.exists(self.abspath):
            return os.path.normpath(self.abspath)

        for prefix, base in searchpaths:
            if self.prefix == prefix:
                path = os.path.join(base, self.relpath)
                if os.path.exists(path):
                    return os.path.normpath(path)
        else:
            return None

    def resolve(self, searchpaths):
        if self.prefix is None and os.path.exists(self.abspath):
            return self
        elif self.prefix is not None:
            for prefix, base in searchpaths:
                if self.prefix == prefix:
                    path = os.path.join(base, self.relpath)
                    if os.path.exists(path):
                        return RecentPath(
                            os.path.normpath(path), self.prefix, self.relpath)
        return None

    @property
    def value(self):
        return os.path.basename(self.abspath)

    @property
    def icon(self):
        provider = QFileIconProvider()
        return provider.icon(QFileIconProvider.Drive)

    @property
    def dirname(self):
        return os.path.dirname(self.abspath)

    def __repr__(self):
        return ("{0.__class__.__name__}(abspath={0.abspath!r}, "
                "prefix={0.prefix!r}, relpath={0.relpath!r}, "
                "title={0.title!r})").format(self)

    __str__ = __repr__


class RecentPathsWidgetMixin:
    SEARCH_PATHS = []

    #: List[RecentPath]
    recent_paths = Setting([])

    _init_called = False

    def __init__(self):
        super().__init__()
        self._init_called = True
        self.relocate_recent_files()

    def _check_init(self):
        if not self._init_called:
            raise RuntimeError("RecentPathsWidgetMixin.__init__ was not called")

    def _search_paths(self):
        basedir = self.workflowEnv().get("basedir", None)
        if basedir is None:
            return self.SEARCH_PATHS
        return self.SEARCH_PATHS + [("basedir", basedir)]

    def relocate_recent_files(self):
        self._check_init()
        search_paths = self._search_paths()
        rec = []
        for recent in self.recent_paths:
            resolved = recent.resolve(search_paths)
            if resolved is not None:
                rec.append(
                    RecentPath.create(resolved.abspath, search_paths))
            elif recent.search(search_paths) is not None:
                rec.append(
                    RecentPath.create(recent.search(search_paths), search_paths)
                )
        self.recent_paths = rec

    def set_file_list(self):
        self._check_init()
        self.file_combo.clear()
        if not self.recent_paths:
            self.file_combo.addItem("(none)")
            self.file_combo.model().item(0).setEnabled(False)
        else:
            for i, recent in enumerate(self.recent_paths):
                self.file_combo.addItem(recent.value)
                self.file_combo.model().item(i).setToolTip(recent.abspath)

    def add_path(self, filename):
        self._check_init()
        recent = RecentPath.create(filename, self._search_paths())
        if recent in self.recent_paths:
            self.recent_paths.remove(recent)
        self.recent_paths.insert(0, recent)

    def workflowEnvChanged(self, key, value, oldvalue):
        if key == "basedir":
            self.relocate_recent_files()
            self.set_file_list()
