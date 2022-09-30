from functools import wraps

from AnyQt.QtCore import Qt, QLocale, QPoint, QT_VERSION_INFO
from AnyQt.QtTest import QTest
from AnyQt.QtGui import QContextMenuEvent
from AnyQt.QtWidgets import QApplication, QWidget, QButtonGroup

from orangewidget.tests.utils import (
    simulate, excepthook_catch, EventSpy, mouseMove
)

from Orange.data import Table, Domain, ContinuousVariable

# pylint: disable=self-assigning-variable,invalid-name
EventSpy = EventSpy
excepthook_catch = excepthook_catch
simulate = simulate
mouseMove = mouseMove


def qbuttongroup_emit_clicked(bg: QButtonGroup, id_: int):
    button = bg.button(id_)
    bg.buttonClicked.emit(button)
    if QT_VERSION_INFO >= (5, 15):
        bg.idClicked.emit(id_)
    if QT_VERSION_INFO < (6, 0):
        bg.buttonClicked[int].emit(id_)


def override_locale(language):
    """Execute the wrapped code with a different locale."""
    def wrapper(f):
        def wrap(*args, **kwargs):
            locale = QLocale()
            QLocale.setDefault(QLocale(language))
            result = f(*args, **kwargs)
            QLocale.setDefault(locale)
            return result
        return wrap
    return wrapper


def contextMenu(
        widget: QWidget, pos=QPoint(), reason=QContextMenuEvent.Mouse,
        modifiers=Qt.NoModifier, delay=-1
) -> None:
    """
    Simulate a context menu event on `widget`.

    `pos` is the event origin specified in widget's local coordinates. If not
    specified. Then widget.rect().center() is used instead.
    """
    if pos.isNull():
        pos = widget.rect().center()
    globalPos = widget.mapToGlobal(pos)
    ev = QContextMenuEvent(reason, pos, globalPos, modifiers)
    if delay >= 0:
        QTest.qWait(delay)
    QApplication.sendEvent(widget, ev)


def table_dense_sparse(test_case):
    # type: (Callable) -> Callable
    """Run a single test case on both dense and sparse Orange tables.

    Examples
    --------
    >>> @table_dense_sparse
    ... def test_something(self, prepare_table):
    ...     data: Table  # The table you want to test on
    ...     data = prepare_table(data)  # This converts the table to dense/sparse

    """

    @wraps(test_case)
    def _wrapper(self):
        # Make sure to call setUp and tearDown methods in between test runs so
        # any widget state doesn't interfere between tests
        test_case(self, lambda table: table.to_dense())
        self.tearDown()
        self.setUp()
        test_case(self, lambda table: table.to_sparse())

    return _wrapper


def possible_duplicate_table(name, table_name='iris', class_var=False):
    """
    Used for checking whether widget resolves possible domain duplicates.
    If the programmer inputs name that will create duplicates and it later fails,
    that's on them.
    """
    data = Table(table_name)
    attributes = data.domain.attributes
    class_vars = list(data.domain.class_vars)
    if class_var:
        class_vars[0] = ContinuousVariable(name)
    else:
        attributes = list(data.domain.attributes[1:])
        attributes.append(ContinuousVariable(name))
    domain = Domain(attributes,
                    class_vars,
                    data.domain.metas)
    return Table.from_numpy(domain, data.X, data.Y, data.metas)
