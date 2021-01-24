import shutil
import tempfile
import uuid

import sys
import os
import tokenize
import unicodedata

from jupyter_client import KernelManager

from typing import Optional, List, TYPE_CHECKING

import pygments.style
from pygments.token import Comment, Keyword, Number, String, Punctuation, Operator, Error, Name
from qtconsole.pygments_highlighter import PygmentsHighlighter
from qtconsole import styles
from qtconsole.client import QtKernelClient
from qtconsole.manager import QtKernelManager


from AnyQt.QtWidgets import (
    QListView, QSizePolicy, QMenu, QSplitter, QLineEdit,
    QAction, QToolButton, QFileDialog, QStyledItemDelegate,
    QStyleOptionViewItem, QPlainTextDocumentLayout,
    QLabel, QWidget, QHBoxLayout, QApplication)
from AnyQt.QtGui import (
    QColor, QBrush, QPalette, QFont, QTextDocument, QTextCharFormat,
    QKeySequence, QFontMetrics, QPainter
)
from AnyQt.QtCore import (
    Qt, QByteArray, QItemSelectionModel, QSize, QRectF, QTimer
)

from orangewidget.widget import Msg

from Orange.data import Table
from Orange.base import Learner, Model
from Orange.widgets import gui
from Orange.widgets.data.utils.python_console import OrangeConsoleWidget
from Orange.widgets.data.utils.pythoneditor.editor import PythonEditor
from Orange.widgets.utils import itemmodels
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output

if TYPE_CHECKING:
    from typing_extensions import TypedDict

__all__ = ["OWPythonScript"]


DEFAULT_SCRIPT = """import numpy as np
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

domain = Domain([ContinuousVariable("age"),
                 ContinuousVariable("height"),
                 DiscreteVariable("gender", values=("M", "F"))])
arr = np.array([
  [25, 186, 0],
  [30, 164, 1]])
out_data = Table.from_numpy(domain, arr)
"""

def text_format(foreground=Qt.black, weight=QFont.Normal):
    fmt = QTextCharFormat()
    fmt.setForeground(QBrush(foreground))
    fmt.setFontWeight(weight)
    return fmt


def read_file_content(filename, limit=None):
    try:
        with open(filename, encoding="utf-8", errors='strict') as f:
            text = f.read(limit)
            return text
    except (OSError, UnicodeDecodeError):
        return None


# pylint: disable=pointless-string-statement
"""
Adapted from jupyter notebook, which was adapted from GitHub.

Highlighting styles are applied with pygments.

pygments does not support partial highlighting; on every character
typed, it performs a full pass of the code. If performance is ever
an issue, revert to prior commit, which uses Qutepart's syntax
highlighting implementation.
"""
SYNTAX_HIGHLIGHTING_STYLES = {
    'Light': {
        Punctuation: "#000",
        Error: '#f00',

        Keyword: 'bold #008000',

        Name: '#212121',
        Name.Function: '#00f',
        Name.Variable: '#05a',
        Name.Decorator: '#aa22ff',
        Name.Builtin: '#008000',
        Name.Builtin.Pseudo: '#05a',

        String: '#ba2121',

        Number: '#080',

        Operator: 'bold #aa22ff',
        Operator.Word: 'bold #008000',

        Comment: 'italic #408080',
    },
    'Dark': {
        Punctuation: "#fff",
        Error: '#f00',

        Keyword: 'bold #4caf50',

        Name: '#e0e0e0',
        Name.Function: '#1e88e5',
        Name.Variable: '#42a5f5',
        Name.Decorator: '#aa22ff',
        Name.Builtin: '#43a047',
        Name.Builtin.Pseudo: '#42a5f5',

        String: '#ff7070',

        Number: '#66bb6a',

        Operator: 'bold #aa22ff',
        Operator.Word: 'bold #4caf50',

        Comment: 'italic #408080',
    }
}


def make_pygments_style(scheme_name):
    """
    Dynamically create a PygmentsStyle class,
    given the name of one of the above highlighting schemes.
    """
    return type(
        'PygmentsStyle',
        (pygments.style.Style,),
        {'styles': SYNTAX_HIGHLIGHTING_STYLES[scheme_name]}
    )


class FakeSignatureMixin:
    def __init__(self, parent, highlighting_scheme, font):
        super().__init__(parent)
        self.highlighting_scheme = highlighting_scheme
        self.setFont(font)
        self.bold_font = QFont(font)
        self.bold_font.setBold(True)

        self.indentation_level = 0

        self._char_4_width = QFontMetrics(font).horizontalAdvance('4444')

    def setIndent(self, margins_width):
        self.setContentsMargins(max(0,
                                    round(margins_width) +
                                    (self.indentation_level - 1 * self._char_4_width)),
                                0, 0, 0)


class FunctionSignature(FakeSignatureMixin, QLabel):
    def __init__(self, parent, highlighting_scheme, font, function_name="python_script"):
        super().__init__(parent, highlighting_scheme, font)
        self.signal_prefix = 'in_'

        # `def python_script(`
        self.prefix = ('<b style="color: ' +
                       self.highlighting_scheme[Keyword].split(' ')[-1] +
                       ';">def </b>'
                       '<span style="color: ' +
                       self.highlighting_scheme[Name.Function].split(' ')[-1] +
                       ';">' + function_name + '</span>'
                       '<span style="color: ' +
                       self.highlighting_scheme[Punctuation].split(' ')[-1] +
                       ';">(</span>')

        # `):`
        self.affix = ('<span style="color: ' +
                      self.highlighting_scheme[Punctuation].split(' ')[-1] +
                      ';">):</span>')

        self.update_signal_text({})

    def update_signal_text(self, signal_values_lengths):
        if not self.signal_prefix:
            return
        lbl_text = self.prefix
        if len(signal_values_lengths) > 0:
            for name, value in signal_values_lengths.items():
                if value == 1:
                    lbl_text += self.signal_prefix + name + ', '
                elif value > 1:
                    lbl_text += self.signal_prefix + name + 's, '
            lbl_text = lbl_text[:-2]  # shave off the trailing ', '
        lbl_text += self.affix
        if self.text() != lbl_text:
            self.setText(lbl_text)
            self.update()


class ReturnStatement(FakeSignatureMixin, QWidget):
    def __init__(self, parent, highlighting_scheme, font):
        super().__init__(parent, highlighting_scheme, font)

        self.indentation_level = 1
        self.signal_labels = {}
        self._prefix = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # `return `
        ret_lbl = QLabel('<b style="color: ' + \
                         highlighting_scheme[Keyword].split(' ')[-1] + \
                         ';">return </b>', self)
        ret_lbl.setFont(self.font())
        ret_lbl.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(ret_lbl)

        # `out_data[, ]` * 4
        self.make_signal_labels('out_')

        layout.addStretch()
        self.setLayout(layout)

    def make_signal_labels(self, prefix):
        self._prefix = prefix
        # `in_data[, ]`
        for i, signal in enumerate(OWPythonScript.signal_names):
            # adding an empty b tag like this adjusts the
            # line height to match the rest of the labels
            signal_display_name = signal
            signal_lbl = QLabel('<b></b>' + prefix + signal_display_name, self)
            signal_lbl.setFont(self.font())
            signal_lbl.setContentsMargins(0, 0, 0, 0)
            self.layout().addWidget(signal_lbl)

            self.signal_labels[signal] = signal_lbl

            if i >= len(OWPythonScript.signal_names) - 1:
                break

            comma_lbl = QLabel(', ')
            comma_lbl.setFont(self.font())
            comma_lbl.setContentsMargins(0, 0, 0, 0)
            comma_lbl.setStyleSheet('.QLabel { color: ' +
                                    self.highlighting_scheme[Punctuation].split(' ')[-1] +
                                    '; }')
            self.layout().addWidget(comma_lbl)

    def update_signal_text(self, signal_name, values_length):
        if not self._prefix:
            return
        lbl = self.signal_labels[signal_name]
        if values_length == 0:
            text = '<b></b>' + self._prefix + signal_name
        else:  # if values_length == 1:
            text = '<b>' + self._prefix + signal_name + '</b>'
        if lbl.text() != text:
            lbl.setText(text)
            lbl.update()


class VimIndicator(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.indicator_color = QColor('#33cc33')
        self.indicator_text = 'normal'

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(self.indicator_color)

        p.save()
        p.setPen(Qt.NoPen)
        fm = QFontMetrics(self.font())
        width = self.rect().width()
        height = fm.height() + 6
        rect = QRectF(0, 0, width, height)
        p.drawRoundedRect(rect, 5, 5)
        p.restore()

        textstart = (width - fm.width(self.indicator_text)) / 2
        p.drawText(textstart, height / 2 + 5, self.indicator_text)

    def minimumSizeHint(self):
        fm = QFontMetrics(self.font())
        width = round(fm.width(self.indicator_text)) + 10
        height = fm.height() + 6
        return QSize(width, height)


class Script:
    Modified = 1
    MissingFromFilesystem = 2

    def __init__(self, name, script, flags=0, filename=None):
        self.name = name
        self.script = script
        self.flags = flags
        self.filename = filename

    def asdict(self) -> '_ScriptData':
        return dict(name=self.name, script=self.script, filename=self.filename)

    @classmethod
    def fromdict(cls, state: '_ScriptData') -> 'Script':
        return Script(state["name"], state["script"], filename=state["filename"])


class ScriptItemDelegate(QStyledItemDelegate):
    # pylint: disable=no-self-use
    def displayText(self, script, _locale):
        if script.flags & Script.Modified:
            return "*" + script.name
        else:
            return script.name

    def paint(self, painter, option, index):
        script = index.data(Qt.DisplayRole)

        if script.flags & Script.Modified:
            option = QStyleOptionViewItem(option)
            option.palette.setColor(QPalette.Text, QColor(Qt.red))
            option.palette.setColor(QPalette.Highlight, QColor(Qt.darkRed))
        super().paint(painter, option, index)

    def createEditor(self, parent, _option, _index):
        return QLineEdit(parent)

    def setEditorData(self, editor, index):
        script = index.data(Qt.DisplayRole)
        editor.setText(script.name)

    def setModelData(self, editor, model, index):
        model[index.row()].name = str(editor.text())


def select_row(view, row):
    """
    Select a `row` in an item view
    """
    selmodel = view.selectionModel()
    selmodel.select(view.model().index(row, 0),
                    QItemSelectionModel.ClearAndSelect)


if TYPE_CHECKING:
    # pylint: disable=used-before-assignment
    _ScriptData = TypedDict("_ScriptData", {
        "name": str, "script": str, "filename": Optional[str]
    })


class OWPythonScript(OWWidget):
    name = "Python Script"
    description = "Write a Python script and run it on input data or models."
    icon = "icons/PythonScript.svg"
    priority = 3150
    keywords = ["program", "function"]

    class Inputs:
        data = Input("Data", Table, replaces=["in_data"],
                     default=True, multiple=True)
        learner = Input("Learner", Learner, replaces=["in_learner"],
                        default=True, multiple=True)
        classifier = Input("Classifier", Model, replaces=["in_classifier"],
                           default=True, multiple=True)
        object = Input("Object", object, replaces=["in_object"],
                       default=False, multiple=True)

    class Outputs:
        data = Output("Data", Table, replaces=["out_data"])
        learner = Output("Learner", Learner, replaces=["out_learner"])
        classifier = Output("Classifier", Model, replaces=["out_classifier"])
        object = Output("Object", object, replaces=["out_object"])

    signal_names = ("data", "learner", "classifier", "object")

    settings_version = 2
    scriptLibrary: 'List[_ScriptData]' = Setting([{
        "name": "Table from numpy",
        "script": DEFAULT_SCRIPT,
        "filename": None
    }])
    currentScriptIndex = Setting(0)
    scriptText: Optional[str] = Setting(None, schema_only=True)
    splitterState: Optional[bytes] = Setting(None)

    vimModeEnabled = Setting(False)

    class Warning(OWWidget.Warning):
        illegal_var_type = Msg('{} should be of type {}, not {}.')

    class Error(OWWidget.Error):
        pass

    def __init__(self):
        super().__init__()

        for name in self.signal_names:
            setattr(self, name, {})

        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)

        # Styling

        self.defaultFont = defaultFont = (
            'Menlo' if sys.platform == 'darwin' else
            'Courier' if sys.platform in ['win32', 'cygwin'] else
            'DejaVu Sans Mono'
        )
        self.defaultFontSize = defaultFontSize = 13

        self.editorBox = gui.vBox(self, box="Editor", spacing=4)
        self.splitCanvas.addWidget(self.editorBox)

        darkMode = QApplication.instance().property('darkMode')
        scheme_name = 'Dark' if darkMode else 'Light'
        syntax_highlighting_scheme = SYNTAX_HIGHLIGHTING_STYLES[scheme_name]
        self.pygments_style_class = make_pygments_style(scheme_name)

        eFont = QFont(defaultFont)
        eFont.setPointSize(defaultFontSize)

        # Fake Signature

        self.func_sig = func_sig = FunctionSignature(
            self.editorBox,
            syntax_highlighting_scheme,
            eFont
        )

        # Editor

        editor = PythonEditor(self)
        editor.setFont(eFont)
        editor.setup_completer_appearance((300, 180), eFont)

        # Fake return

        return_stmt = ReturnStatement(
            self.editorBox,
            syntax_highlighting_scheme,
            eFont
        )
        self.return_stmt = return_stmt

        # Match indentation
        textEditBox = QWidget(self.editorBox)
        textEditBox.setLayout(QHBoxLayout())
        char_4_width = QFontMetrics(eFont).horizontalAdvance('0000')

        @editor.viewport_margins_updated.connect
        def _(width):
            func_sig.setIndent(width)
            textEditMargin = max(0, round(char_4_width - width))
            return_stmt.setIndent(textEditMargin + width)
            textEditBox.layout().setContentsMargins(
                int(textEditMargin), 0, 0, 0
            )

        self.editor = editor
        textEditBox.layout().addWidget(editor)
        self.editorBox.layout().addWidget(func_sig)
        self.editorBox.layout().addWidget(textEditBox)
        self.editorBox.layout().addWidget(return_stmt)

        self.editorBox.setAlignment(Qt.AlignVCenter)
        self.editor.setTabStopWidth(4)

        self.editor.modificationChanged[bool].connect(self.onModificationChanged)

        # Console

        self.consoleBox = gui.vBox(self, 'Console')
        self.splitCanvas.addWidget(self.consoleBox)

        # Qtconsole

        jupyter_widget = OrangeConsoleWidget(
            style_sheet=styles.default_light_style_sheet
        )
        jupyter_widget.results_ready.connect(self.receive_outputs)

        jupyter_widget._highlighter.set_style(self.pygments_style_class)
        jupyter_widget.font_family = defaultFont
        jupyter_widget.font_size = defaultFontSize
        jupyter_widget.reset_font()

        self.console = jupyter_widget
        self.consoleBox.layout().addWidget(self.console)
        self.consoleBox.setAlignment(Qt.AlignBottom)
        self.setAcceptDrops(True)

        self.statuses = []

        # 'Injecting variables...' is set in handleNewVars

        @self.console.variables_finished_injecting.connect
        def _():
            self.clear_status('Injecting variables...')

        @self.console.begun_collecting_variables.connect
        def _():
            self.set_status('Collecting variables...')

        # 'Collecting variables...' is reset in receive_outputs

        @self.console.execution_started.connect
        def _():
            self.set_status('Running script...', force=True)
            # trigger console repaint
            # (for some reason repaint is broken if not singleShotting)
            QTimer.singleShot(0, self.console.update)

        @self.console.execution_finished.connect
        def _():
            self.clear_status('Running script...')
            # trigger console repaint
            QTimer.singleShot(0, self.console.update)

        # Kernel stuff

        self.kernel_client: QtKernelClient = None
        self.kernel_manager: KernelManager = None
        self.init_kernel()

        # Controls

        self.editor_controls = gui.vBox(self.controlArea, box='Preferences')

        # Vim

        self.vim_box = gui.hBox(self.editor_controls, spacing=20)
        self.vim_indicator = VimIndicator(self.vim_box)

        vim_sp = QSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        vim_sp.setRetainSizeWhenHidden(True)
        self.vim_indicator.setSizePolicy(vim_sp)

        def enable_vim_mode():
            editor.vimModeEnabled = self.vimModeEnabled
            self.vim_indicator.setVisible(self.vimModeEnabled)
        enable_vim_mode()

        gui.checkBox(
            self.vim_box, self, 'vimModeEnabled', 'Vim mode',
            tooltip="Only for the coolest.",
            callback=enable_vim_mode
        )
        self.vim_box.layout().addWidget(self.vim_indicator)
        @editor.vimModeIndicationChanged.connect
        def _(color, text):
            self.vim_indicator.indicator_color = color
            self.vim_indicator.indicator_text = text
            self.vim_indicator.update()

        # Library

        self.libraryListSource = []
        self._cachedDocuments = {}

        self.libraryList = itemmodels.PyListModel(
            [], self,
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)

        self.libraryList.wrap(self.libraryListSource)

        self.controlBox = gui.vBox(self.controlArea, 'Library')
        self.controlBox.layout().setSpacing(1)

        self.libraryView = QListView(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored,
                                   QSizePolicy.Preferred)
        )
        self.libraryView.setItemDelegate(ScriptItemDelegate(self))
        self.libraryView.setModel(self.libraryList)

        self.libraryView.selectionModel().selectionChanged.connect(
            self.onSelectedScriptChanged
        )
        self.controlBox.layout().addWidget(self.libraryView)

        w = itemmodels.ModelActionsWidget()

        self.addNewScriptAction = action = QAction("+", self)
        action.setToolTip("Add a new script to the library")
        action.triggered.connect(self.onAddScript)
        w.addAction(action)

        action = QAction(unicodedata.lookup("MINUS SIGN"), self)
        action.setToolTip("Remove script from library")
        action.triggered.connect(self.onRemoveScript)
        w.addAction(action)

        action = QAction("Update", self)
        action.setToolTip("Save changes in the editor to library")
        action.setShortcut(QKeySequence(QKeySequence.Save))
        action.triggered.connect(self.commitChangesToLibrary)
        w.addAction(action)

        action = QAction("More", self, toolTip="More actions")

        new_from_file = QAction("Import Script from File", self)
        save_to_file = QAction("Save Selected Script to File", self)
        restore_saved = QAction("Undo Changes to Selected Script", self)
        save_to_file.setShortcut(QKeySequence(QKeySequence.SaveAs))

        new_from_file.triggered.connect(self.onAddScriptFromFile)
        save_to_file.triggered.connect(self.saveScript)
        restore_saved.triggered.connect(self.restoreSaved)

        menu = QMenu(w)
        menu.addAction(new_from_file)
        menu.addAction(save_to_file)
        menu.addAction(restore_saved)
        action.setMenu(menu)
        button = w.addAction(action)
        button.setPopupMode(QToolButton.InstantPopup)

        w.layout().setSpacing(1)

        self.controlBox.layout().addWidget(w)

        self.execute_button = gui.button(self.buttonsArea, self, 'Run', callback=self.commit)

        self.run_action = QAction("Run script", self, triggered=self.commit,
                                  shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_R))
        self.addAction(self.run_action)

        self.saveAction = action = QAction("&Save", self.editor)
        action.setToolTip("Save script to file")
        action.setShortcut(QKeySequence(QKeySequence.Save))
        action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(self.saveScript)

        # And finally,

        self.splitCanvas.setSizes([2, 1])
        self._restoreState()
        self.settingsAboutToBePacked.connect(self._saveState)

    def sizeHint(self) -> QSize:
        return super().sizeHint().expandedTo(QSize(810, 600))

    def _restoreState(self):
        self.libraryListSource = [Script.fromdict(s) for s in self.scriptLibrary]
        self.libraryList.wrap(self.libraryListSource)
        select_row(self.libraryView, self.currentScriptIndex)

        if self.scriptText is not None:
            current = self.editor.toPlainText()
            # do not mark scripts as modified
            if self.scriptText != current:
                self.editor.document().setPlainText(self.scriptText)

        if self.splitterState is not None:
            self.splitCanvas.restoreState(QByteArray(self.splitterState))

    def init_kernel(self):
        if self.kernel_manager is not None:
            self.shutdown_kernel()

        self._temp_connection_dir = tempfile.mkdtemp()

        ident = str(uuid.uuid4()).split('-')[-1]
        cf = os.path.join(self._temp_connection_dir, 'kernel-%s.json' % ident)

        self.kernel_manager = QtKernelManager(
            connection_file=cf
        )

        self.kernel_manager.start_kernel(
            extra_arguments=[
                '--IPKernelApp.kernel_class='
                'Orange.widgets.data.utils.python_kernel.OrangeIPythonKernel',
                '--matplotlib='
                'inline'
            ]
        )
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        if self.editor is not None:
            self.editor.kernel_manager = self.kernel_manager
            self.editor.kernel_client = self.kernel_client
        if self.console is not None:
            self.console.kernel_manager = self.kernel_manager
            self.console.kernel_client = self.kernel_client
            self.console.set_kernel_id(ident)

    def shutdown_kernel(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()
        shutil.rmtree(self._temp_connection_dir)

    def _saveState(self):
        self.scriptLibrary = [s.asdict() for s in self.libraryListSource]
        self.scriptText = self.editor.toPlainText()
        self.splitterState = bytes(self.splitCanvas.saveState())

    def handle_input(self, obj, sig_id, signal):
        dic = getattr(self, signal)
        if obj is None:
            if sig_id in dic.keys():
                del dic[sig_id]
        else:
            dic[sig_id] = obj

    def clear_status(self, msg):
        if msg not in self.statuses:
            return
        self.statuses.remove(msg)
        self.__update_status()

    def set_status(self, msg, force=False):
        if msg in self.statuses:
            if force:
                self.statuses.remove(msg)
                self.statuses.insert(0, msg)
            return
        if force:
            self.statuses.insert(0, msg)
        else:
            self.statuses.append(msg)
        self.__update_status()

    def __update_status(self):
        if self.statuses:
            msg = self.statuses[0]
        else:
            msg = ''

        self.setStatusMessage(msg)

    def receive_outputs(self, out_vars):
        self.clear_status('Collecting variables...')
        self.progressBar()
        for signal in self.signal_names:
            out_name = "out_" + signal
            req_type = self.Outputs.__dict__[signal].type

            output = getattr(self.Outputs, signal)
            if out_name not in out_vars:
                output.send(None)
                continue
            var = out_vars[out_name]

            if not isinstance(var, req_type):
                output.send(None)
                actual_type = type(var)
                self.Warning.illegal_var_type(out_name,
                                              req_type.__module__ + '.' + req_type.__name__,
                                              actual_type.__module__ + '.' + actual_type.__name__)
                continue

            output.send(var)

    @Inputs.data
    def set_data(self, data, sig_id):
        self.handle_input(data, sig_id, "data")

    @Inputs.learner
    def set_learner(self, data, sig_id):
        self.handle_input(data, sig_id, "learner")

    @Inputs.classifier
    def set_classifier(self, data, sig_id):
        self.handle_input(data, sig_id, "classifier")

    @Inputs.object
    def set_object(self, data, sig_id):
        self.handle_input(data, sig_id, "object")

    def handleNewSignals(self):
        # update fake signature labels
        self.func_sig.update_signal_text({
            n: len(getattr(self, n)) for n in self.signal_names
        })

        self.set_status('Injecting variables...')
        vars = self.initial_locals_state()
        self.console.set_vars(vars)

    def selectedScriptIndex(self):
        rows = self.libraryView.selectionModel().selectedRows()
        if rows:
            return [i.row() for i in rows][0]
        else:
            return None

    def setSelectedScript(self, index):
        select_row(self.libraryView, index)

    def onAddScript(self, *_):
        self.libraryList.append(Script("New script", self.editor.toPlainText(), 0))
        self.setSelectedScript(len(self.libraryList) - 1)

    def onAddScriptFromFile(self, *_):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Python Script',
            os.path.expanduser("~/"),
            'Python files (*.py)\nAll files(*.*)'
        )
        if filename:
            name = os.path.basename(filename)
            with tokenize.open(filename) as f:
                contents = f.read()
            self.libraryList.append(Script(name, contents, 0, filename))
            self.setSelectedScript(len(self.libraryList) - 1)

    def onRemoveScript(self, *_):
        index = self.selectedScriptIndex()
        if index is not None:
            del self.libraryList[index]
            select_row(self.libraryView, max(index - 1, 0))

    def onSaveScriptToFile(self, *_):
        index = self.selectedScriptIndex()
        if index is not None:
            self.saveScript()

    def onSelectedScriptChanged(self, selected, _deselected):
        index = [i.row() for i in selected.indexes()]
        if index:
            current = index[0]
            if current >= len(self.libraryList):
                self.addNewScriptAction.trigger()
                return

            self.editor.setDocument(self.documentForScript(current))
            self.currentScriptIndex = current

    def documentForScript(self, script=0):
        if not isinstance(script, Script):
            script = self.libraryList[script]
        if script not in self._cachedDocuments:
            doc = QTextDocument(self)
            doc.setDocumentLayout(QPlainTextDocumentLayout(doc))
            doc.setPlainText(script.script)
            doc.setDefaultFont(QFont(self.defaultFont))
            doc.highlighter = PygmentsHighlighter(doc)
            doc.highlighter.set_style(self.pygments_style_class)
            doc.setDefaultFont(QFont(self.defaultFont, pointSize=self.defaultFontSize))
            doc.modificationChanged[bool].connect(self.onModificationChanged)
            doc.setModified(False)
            self._cachedDocuments[script] = doc
        return self._cachedDocuments[script]

    def commitChangesToLibrary(self, *_):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index].script = self.editor.toPlainText()
            self.editor.document().setModified(False)
            self.libraryList.emitDataChanged(index)

    def onModificationChanged(self, modified):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index].flags = Script.Modified if modified else 0
            self.libraryList.emitDataChanged(index)

    def restoreSaved(self):
        index = self.selectedScriptIndex()
        if index is not None:
            self.editor.document().setPlainText(self.libraryList[index].script)
            self.editor.document().setModified(False)

    def saveScript(self):
        index = self.selectedScriptIndex()
        if index is not None:
            script = self.libraryList[index]
            filename = script.filename
        else:
            filename = os.path.expanduser("~/")

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Python Script',
            filename,
            'Python files (*.py)\nAll files(*.*)'
        )

        if filename:
            fn = ""
            head, tail = os.path.splitext(filename)
            if not tail:
                fn = head + ".py"
            else:
                fn = filename

            f = open(fn, 'w')
            f.write(self.editor.toPlainText())
            f.close()

    def initial_locals_state(self):
        d = {}
        for name in self.signal_names:
            value = getattr(self, name)
            all_values = list(value.values())
            d[name + "s"] = all_values
        return d

    def commit(self):
        self.Warning.clear()
        self.Error.clear()

        script = str(self.editor.text)
        self.console.run_script(script)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.InsertLineSeparator):
            # run on Shift+Enter, Ctrl+Enter
            self.run_action.trigger()
            event.accept()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):  # pylint: disable=no-self-use
        urls = event.mimeData().urls()
        if urls:
            # try reading the file as text
            c = read_file_content(urls[0].toLocalFile(), limit=1000)
            if c is not None:
                event.acceptProposedAction()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is not None and version < 2:
            scripts = settings.pop("libraryListSource")  # type: List[Script]
            library = [dict(name=s.name, script=s.script, filename=s.filename)
                       for s in scripts]  # type: List[_ScriptData]
            settings["scriptLibrary"] = library

    def onDeleteWidget(self):
        self.text.terminate()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPythonScript).run()
