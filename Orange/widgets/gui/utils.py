from types import LambdaType

from AnyQt.QtWidgets import QApplication, QWidget, QSizePolicy
from AnyQt.QtGui import QCursor
from AnyQt.QtCore import Qt

from Orange.util import namegen
from Orange.widgets.utils.buttons import VariableTextPushButton

from .boxes import widgetBox
from .checkbox import checkBox
from .base import is_horizontal, miscellanea

__all__ = ["auto_commit", "ProgressBar"]


LAMBDA_NAME = namegen('_lambda_')


def auto_commit(widget, master, value, label, auto_label=None, box=True,
                checkbox_label=None, orientation=None, commit=None,
                callback=None, **misc):
    """
    Add a commit button with auto-commit check box.

    The widget must have a commit method and a setting that stores whether
    auto-commit is on.

    The function replaces the commit method with a new commit method that
    checks whether auto-commit is on. If it is, it passes the call to the
    original commit, otherwise it sets the dirty flag.

    The checkbox controls the auto-commit. When auto-commit is switched on, the
    checkbox callback checks whether the dirty flag is on and calls the original
    commit.

    Important! Do not connect any signals to the commit before calling
    auto_commit.

    :param widget: the widget into which the box with the button is inserted
    :type widget: QWidget or None
    :param value: the master's attribute which stores whether the auto-commit
        is on
    :type value:  str
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param label: The button label
    :type label: str
    :param auto_label: The label used when auto-commit is on; default is
        `label + " Automatically"`
    :type auto_label: str
    :param commit: master's method to override ('commit' by default)
    :type commit: function
    :param callback: function to call whenever the checkbox's statechanged
    :type callback: function
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :return: the box
    """
    def checkbox_toggled():
        if getattr(master, value):
            btn.setText(auto_label)
            btn.setEnabled(False)
            if dirty:
                do_commit()
        else:
            btn.setText(label)
            btn.setEnabled(True)
        if callback:
            callback()

    def unconditional_commit():
        nonlocal dirty
        if getattr(master, value):
            do_commit()
        else:
            dirty = True

    def do_commit():
        nonlocal dirty
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            commit()
            dirty = False
        finally:
            QApplication.restoreOverrideCursor()

    dirty = False
    commit = commit or getattr(master, 'commit')
    commit_name = next(LAMBDA_NAME) if isinstance(commit, LambdaType) else commit.__name__
    setattr(master, 'unconditional_' + commit_name, commit)

    if not auto_label:
        if checkbox_label:
            auto_label = label
        else:
            auto_label = label.title() + " Automatically"
    if isinstance(box, QWidget):
        b = box
    else:
        if orientation is None:
            orientation = Qt.Vertical if checkbox_label else Qt.Horizontal
        b = widgetBox(widget, box=box, orientation=orientation,
                      addToLayout=False)
        b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    b.checkbox = cb = checkBox(b, master, value, checkbox_label,
                               callback=checkbox_toggled, tooltip=auto_label)
    if is_horizontal(orientation):
        b.layout().addSpacing(10)
    cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    b.button = btn = VariableTextPushButton(
        b, text=label, textChoiceList=[label, auto_label], clicked=do_commit)
    if b.layout() is not None:
        b.layout().addWidget(b.button)

    if not checkbox_label:
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    checkbox_toggled()
    setattr(master, commit_name, unconditional_commit)
    misc['addToLayout'] = misc.get('addToLayout', True) and \
                          not isinstance(box, QWidget)
    miscellanea(b, widget, widget, **misc)
    return b


class ProgressBar:
    def __init__(self, widget, iterations):
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()
        self.finished = False

    def __del__(self):
        if not self.finished:
            self.widget.progressBarFinished(processEvents=False)

    def advance(self, count=1):
        self.count += count
        self.widget.progressBarSet(int(self.count * 100 / max(1, self.iter)))

    def finish(self):
        self.finished = True
        self.widget.progressBarFinished()
