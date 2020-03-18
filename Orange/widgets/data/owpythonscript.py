import sys
import os
import code
import keyword
import itertools
import tokenize
import unicodedata
import codecs
import pickle
from functools import reduce
from collections import defaultdict
from unittest.mock import patch

from typing import Optional, List, TYPE_CHECKING, NamedTuple

import qutepart

from AnyQt.QtWidgets import (
    QPlainTextEdit, QListView, QSizePolicy, QMenu, QSplitter, QLineEdit,
    QAction, QToolButton, QFileDialog, QStyledItemDelegate,
    QStyleOptionViewItem, QPlainTextDocumentLayout
)
from AnyQt.QtGui import (
    QColor, QBrush, QPalette, QFont, QTextDocument,
    QSyntaxHighlighter, QTextCharFormat, QTextCursor, QKeySequence,
)
from AnyQt.QtCore import Qt, QRegExp, QByteArray, QItemSelectionModel, QSize, Signal

import pygments.style
from pygments.token import Comment, Keyword, Number, String, Punctuation, Operator, Error, Name, Other
from qtconsole import styles
from qtconsole.pygments_highlighter import PygmentsHighlighter
from qtconsole.manager import QtKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from ipython_genutils.tempdir import TemporaryDirectory

from Orange.data import Table
from Orange.base import Learner, Model
from Orange.util import interleave
from Orange.widgets import gui
from Orange.widgets.utils import itemmodels
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output

if TYPE_CHECKING:
    from typing_extensions import TypedDict

__all__ = ["OWPythonScript"]


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
        # TODO
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


PygmentsStyle = make_pygments_style('Light')


def read_file_content(filename, limit=None):
    try:
        with open(filename, encoding="utf-8", errors='strict') as f:
            text = f.read(limit)
            return text
    except (OSError, UnicodeDecodeError):
        return None


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


def get_default_scripts():
    return [{
        "name": "Pandas example",
        "filename": None,
        "script": """\
from Orange.data.pandas_compat import table_to_frame, table_from_frame

# Convert in_data: Orange.data.Table to pandas.DataFrame
in_df = table_to_frame(in_data)
# To include meta attributes:
# in_df = table_to_frame(in_data, include_metas=True)


print('Hello world')
out_df = in_df


# Convert out_df: pandas.DataFrame to Orange.data.Table
out_data = table_from_frame(out_df)
# To interpret strings as discrete attributes:
# out_data = table_from_frame(out_df, force_nominal=True)

# Consider using Select Columns to set out_data features as target/meta\
"""
    }]


class ConsoleWidget(RichJupyterWidget):
    becomes_ready = Signal()

    execution_finished = Signal()

    results_ready = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__queued_execution = None
        self.__prompt_num = 1
        self.__default_in_prompt = self.in_prompt
        self.__executing = False
        self.__is_ready = False

        self.inject_vars_comm = None
        self.collect_vars_comm = None

        self.style_sheet = styles.default_light_style_sheet + \
                           '.run-prompt { color: #aa22ff; }'

        """
        Let the widget/kernel start up before trying to run a script,
        by storing a queued execution payload when the widget's commit
        method is invoked before <In [0]:> appears.
        """
        @self.becomes_ready.connect
        def _():
            self.becomes_ready.disconnect(_)  # reset callback
            self.__initialize_comms()
            self.becomes_ready.connect(self.__on_ready)
            self.__on_ready()

    def __initialize_comms(self):
        self.inject_vars_comm = self.kernel_client.comm_manager.new_comm(
            'inject_vars', {}
        )
        self.collect_vars_comm = self.kernel_client.comm_manager.new_comm(
            'collect_vars', {}
        )
        self.collect_vars_comm.on_msg(self.__on_done)
        self.execution_finished.connect(
            lambda: self.collect_vars_comm.send({})
        )

        def err():
            raise ConnectionAbortedError("Kernel closed run_script comm channel.")
        self.inject_vars_comm.on_close(err)
        self.collect_vars_comm.on_close(err)

    def __on_ready(self):
        self.__is_ready = True
        self.__run_queued_payload()

    def __run_queued_payload(self):
        if self.__queued_execution is None:
            return
        qe = self.__queued_execution
        self.__queued_execution = None
        self.run_script_with_locals(*qe)

    def run_script_with_locals(self, script, locals):
        """
        Inject the in vars, run the script,
        collect the out vars (emit the results_ready signal).
        """
        if not self.__is_ready:
            self.__queued_execution = (script, locals)
            return

        if self.__executing:
            if not self.__queued_execution:
                @self.execution_finished.connect
                def _():
                    self.execution_finished.disconnect(_)  # reset callback
                    self.__run_queued_payload()
            self.__queued_execution = (script, locals)
            self.__is_ready = False
            self.interrupt_kernel()
            return

        @self.inject_vars_comm.on_msg
        def _(msg):
            self.inject_vars_comm.on_msg(None)  # reset callback
            self.__on_variables_injected(msg, script)

        # pickle-strings aren't json-serializable,
        # but with a little bit of magic (and spatial inefficiency)...
        self.inject_vars_comm.send({'locals': {
            k: codecs.encode(pickle.dumps(l), 'base64').decode()
            for k, l in locals.items()
        }})

    def __on_variables_injected(self, msg, script):
        # update prompts
        self._set_input_buffer('')
        self.in_prompt = '<span class="run-prompt">Run[<span class="in-prompt-number">%i</span>]</span>'
        self._update_prompt(self.__prompt_num)
        self._append_plain_text('\n')
        self.in_prompt = 'Running script...'
        self._show_interpreter_prompt(self.__prompt_num)

        # run the script
        self.__executing = True
        # we abuse this method instead of the public ones to keep
        # the 'Running script...' prompt at the bottom of the console
        self.kernel_client.execute(script)

    def __on_done(self, msg):
        data = msg['content']['data']
        outputs = data['outputs']

        out_vars = {
            k: pickle.loads(codecs.decode(l.encode(), 'base64'))
            for k, l in outputs.items()
        }
        self.results_ready.emit(out_vars)

    # override
    def _handle_execute_result(self, msg):
        super()._handle_execute_result(msg)
        if self.__executing:
            self._append_plain_text('\n', before_prompt=True)

    # override
    def _handle_execute_reply(self, msg):
        self.__prompt_num = msg['content']['execution_count'] + 1

        if not self.__executing:
            return super()._handle_execute_reply(msg)
        self.__executing = False

        self.in_prompt = self.__default_in_prompt

        if msg['content']['status'] != 'ok':
            self._show_interpreter_prompt(self.__prompt_num)
            return super()._handle_execute_reply(msg)

        self._update_prompt(self.__prompt_num)

        self.execution_finished.emit()

    # override
    def _handle_kernel_died(self, since_last_heartbeat):
        super()._handle_kernel_died(since_last_heartbeat)
        self.__is_ready = False

    # override
    def _show_interpreter_prompt(self, number=None):
        """
        The console's ready when the first prompt shows up.
        """
        super()._show_interpreter_prompt(number)
        if number is not None and not self.__is_ready:
            self.becomes_ready.emit()

    # override
    def _event_filter_console_keypress(self, event):
        """
        KeyboardInterrupt on run script.
        """
        if self._control_key_down(event.modifiers(), include_command=False) and \
                event.key() == Qt.Key_C and \
                self.__executing:
            self.interrupt_kernel()
            return True
        return super()._event_filter_console_keypress(event)


def make_custom_ipython_kernel_manager():
    """
    Install Orange IPython kernel,
    start a kernel manager instance,
    uninstall Orange IPython kernel
    """
    import json
    import pkg_resources
    from jupyter_client.kernelspec import KernelSpecManager

    # to run our custom kernel (./utils/ipython_kernel),
    # we install it to sys.prefix's (venv) jupyter kernels,
    kernel_json = {
        "argv": [
            sys.executable,
            pkg_resources.resource_filename('Orange', 'widgets/data/utils/ipython_kernel/'),
            '-f',
            '{connection_file}'
        ],
        'display_name': 'orangeipython',
        'language': 'python3',
    }
    ksm = KernelSpecManager()
    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'kernel.json'), 'w') as f:
            json.dump(kernel_json, f, sort_keys=True)
        ksm.install_kernel_spec(td, 'orangeipython', prefix=sys.prefix)

    kernel_manager = QtKernelManager(kernel_name='orangeipython')
    kernel_manager.start_kernel()
    # and clean up behind ourselves by removing it after the kernel starts
    ksm.remove_kernel_spec('orangeipython')

    return kernel_manager


class OWPythonScript(OWWidget):
    name = "Python Script"
    description = "Write a Python script and run it on input data or models."
    icon = "icons/PythonScript.svg"
    priority = 3150
    keywords = ["program"]

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
    scriptLibrary: 'List[_ScriptData]' = Setting(get_default_scripts())
    currentScriptIndex = Setting(0)
    scriptText: Optional[str] = Setting(None, schema_only=True)
    splitterState: Optional[bytes] = Setting(None)

    # Widgets in the same schema share namespace through a dictionary whose
    # key is self.signalManager. ales-erjavec expressed concern (and I fully
    # agree!) about widget being aware of the outside world. I am leaving this
    # anyway. If this causes any problems in the future, replace this with
    # shared_namespaces = {} and thus use a common namespace for all instances
    # of # PythonScript even if they are in different schemata.
    shared_namespaces = defaultdict(dict)

    kernel_manager = None

    class Error(OWWidget.Error):
        pass

    def __init__(self):
        super().__init__()
        if self.kernel_manager is None:
            self.kernel_manager = make_custom_ipython_kernel_manager()
        self.libraryListSource = []

        for name in self.signal_names:
            setattr(self, name, {})

        self._cachedDocuments = {}

        self.infoBox = gui.vBox(self.controlArea, 'Info')
        gui.label(
            self.infoBox, self,
            "<p>Execute python script.</p><p>Input variables:<ul><li> " +
            "<li>".join(map("in_{0}, in_{0}s".format, self.signal_names)) +
            "</ul></p><p>Output variables:<ul><li>" +
            "<li>".join(map("out_{0}".format, self.signal_names)) +
            "</ul></p>"
        )

        self.libraryList = itemmodels.PyListModel(
            [], self,
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)

        self.libraryList.wrap(self.libraryListSource)

        self.controlBox = gui.vBox(self.controlArea, 'Library')
        self.controlBox.layout().setSpacing(1)

        self.libraryView = QListView(
            editTriggers=QListView.DoubleClicked |
                         QListView.EditKeyPressed,
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

        self.execute_button = gui.button(self.controlArea, self, 'Run', callback=self.commit)

        run = QAction("Run script", self, triggered=self.commit,
                      shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_R))
        self.addAction(run)

        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)

        self.defaultFont = defaultFont = 'Menlo'
        self.defaultFontSize = defaultFontSize = 13

        self.textBox = gui.vBox(self, 'Python Script')
        self.splitCanvas.addWidget(self.textBox)

        editor = qutepart.Qutepart(self)

        eFont = QFont(defaultFont)
        eFont.setPointSize(defaultFontSize)
        editor.setFont(eFont)

        # use python autoindent, autocomplete
        editor.detectSyntax(language='Python')
        # but clear the highlighting and use our own
        editor.clearSyntax()
        doc = editor.document()
        highlighter = PygmentsHighlighter(doc)
        highlighter.set_style(PygmentsStyle)
        doc.highlighter = highlighter

        # TODO should we care about displaying the these warnings?
        # editor.userWarning.connect()

        self.editor = editor
        self.editor.modificationChanged[bool].connect(self.onModificationChanged)
        self.textBox.layout().addWidget(self.editor)

        self.textBox.setAlignment(Qt.AlignVCenter)
        self.saveAction = action = QAction("&Save", self.editor)
        action.setToolTip("Save script to file")
        action.setShortcut(QKeySequence(QKeySequence.Save))
        action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(self.saveScript)

        self.consoleBox = gui.vBox(self, 'Console')
        self.splitCanvas.addWidget(self.consoleBox)

        kernel_client = self.kernel_manager.client()
        kernel_client.start_channels()

        jupyter_widget = ConsoleWidget()
        jupyter_widget.results_ready.connect(self.receive_outputs)

        jupyter_widget.kernel_manager = kernel_manager
        jupyter_widget.kernel_client = kernel_client

        jupyter_widget._highlighter.set_style(PygmentsStyle)
        jupyter_widget.font_family = defaultFont
        jupyter_widget.font_size = defaultFontSize
        jupyter_widget.reset_font()

        self.console = jupyter_widget
        self.consoleBox.layout().addWidget(self.console)
        self.consoleBox.setAlignment(Qt.AlignBottom)
        self.splitCanvas.setSizes([2, 1])
        self.setAcceptDrops(True)
        self.controlArea.layout().addStretch(10)

        self._restoreState()
        self.settingsAboutToBePacked.connect(self._saveState)

    def sizeHint(self) -> QSize:
        return super().sizeHint().expandedTo(QSize(800, 600))

    def _restoreState(self):
        self.libraryListSource = [Script.fromdict(s) for s in self.scriptLibrary]
        self.libraryList.wrap(self.libraryListSource)
        select_row(self.libraryView, self.currentScriptIndex)

        if self.scriptText is not None:
            current = self.editor.text
            # do not mark scripts as modified
            if self.scriptText != current:
                self.editor.setText(self.scriptText)

        if self.splitterState is not None:
            self.splitCanvas.restoreState(QByteArray(self.splitterState))

    def _saveState(self):
        self.scriptLibrary = [s.asdict() for s in self.libraryListSource]
        self.scriptText = self.editor.text
        self.splitterState = bytes(self.splitCanvas.saveState())

    def handle_input(self, obj, sig_id, signal):
        sig_id = sig_id[0]
        dic = getattr(self, signal)
        if obj is None:
            if sig_id in dic.keys():
                del dic[sig_id]
        else:
            dic[sig_id] = obj

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
        self.commit()

    def selectedScriptIndex(self):
        rows = self.libraryView.selectionModel().selectedRows()
        if rows:
            return [i.row() for i in rows][0]
        else:
            return None

    def setSelectedScript(self, index):
        select_row(self.libraryView, index)

    def onAddScript(self, *_):
        self.libraryList.append(Script("New script", self.editor.text, 0))
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

            self.editor.text = self.documentForScript(current).toPlainText()
            self.currentScriptIndex = current

    def documentForScript(self, script=0):
        if not isinstance(script, Script):
            script = self.libraryList[script]
        if script not in self._cachedDocuments:
            doc = QTextDocument(self)
            doc.setDocumentLayout(QPlainTextDocumentLayout(doc))
            doc.setPlainText(script.script)
            doc.setDefaultFont(QFont(self.defaultFont))
            doc.modificationChanged[bool].connect(self.onModificationChanged)
            doc.setModified(False)
            self._cachedDocuments[script] = doc
        return self._cachedDocuments[script]

    def commitChangesToLibrary(self, *_):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index].script = self.editor.text
            self.editor.setModified(False)
            self.libraryList.emitDataChanged(index)

    def onModificationChanged(self, modified):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index].flags = Script.Modified if modified else 0
            self.libraryList.emitDataChanged(index)

    def restoreSaved(self):
        index = self.selectedScriptIndex()
        if index is not None:
            self.editor.text = self.libraryList[index].script
            self.editor.setModified(False)

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
            f.write(self.editor.text)
            f.close()

    def initial_locals_state(self):
        d = self.shared_namespaces[self.signalManager].copy()
        for name in self.signal_names:
            value = getattr(self, name)
            if len(value) == 0:
                continue
            all_values = list(value.values())
            one_value = all_values[0] if len(all_values) == 1 else None
            d["in_" + name + "s"] = all_values
            d["in_" + name] = one_value
        return d

    def update_namespace(self, namespace):
        not_saved = reduce(set.union,
                           ({f"in_{name}s", f"in_{name}", f"out_{name}"}
                            for name in self.signal_names))
        self.shared_namespaces[self.signalManager].update(
            {name: value for name, value in namespace.items()
             if name not in not_saved})

    def commit(self):
        script = str(self.editor.text)

        self.console.run_script_with_locals(script, self.initial_locals_state())

    def receive_outputs(self, out_vars):
        for signal in self.signal_names:
            out_name = "out_" + signal
            if out_name not in out_vars:
                continue

            getattr(self.Outputs, signal).send(out_vars[out_name])

    def dragEnterEvent(self, event):  # pylint: disable=no-self-use
        urls = event.mimeData().urls()
        if urls:
            # try reading the file as text
            c = read_file_content(urls[0].toLocalFile(), limit=1000)
            if c is not None:
                event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle file drops"""
        urls = event.mimeData().urls()
        if urls:
            # TODO
            self.editor.pasteFile(urls[0])

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is not None and version < 2:
            scripts = settings.pop("libraryListSource")  # type: List[Script]
            library = [dict(name=s.name, script=s.script, filename=s.filename)
                       for s in scripts]  # type: List[_ScriptData]
            settings["scriptLibrary"] = library


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPythonScript).run()
