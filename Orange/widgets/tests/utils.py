import sys
import warnings
import contextlib

from AnyQt.QtCore import Qt, QObject, QEventLoop, QTimer, QLocale
from AnyQt.QtTest import QTest


class EventSpy(QObject):
    """
    A testing utility class (similar to QSignalSpy) to record events
    delivered to a QObject instance.

    Note
    ----
    Only event types can be recorded (as QEvent instances are deleted
    on delivery).

    Note
    ----
    Can only be used with a QCoreApplication running.

    Parameters
    ----------
    object : QObject
        An object whose events need to be recorded.
    etype : Union[QEvent.Type, Sequence[QEvent.Type]
        A event type (or types) that should be recorded
    """
    def __init__(self, object, etype, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(object, QObject):
            raise TypeError

        self.__object = object
        try:
            len(etype)
        except TypeError:
            etypes = {etype}
        else:
            etypes = set(etype)

        self.__etypes = etypes
        self.__record = []
        self.__loop = QEventLoop()
        self.__timer = QTimer(self, singleShot=True)
        self.__timer.timeout.connect(self.__loop.quit)
        self.__object.installEventFilter(self)

    def wait(self, timeout=5000):
        """
        Start an event loop that runs until a spied event or a timeout occurred.

        Parameters
        ----------
        timeout : int
            Timeout in milliseconds.

        Returns
        -------
        res : bool
            True if the event occurred and False otherwise.

        Example
        -------
        >>> app = QCoreApplication.instance() or QCoreApplication([])
        >>> obj = QObject()
        >>> spy = EventSpy(obj, QEvent.User)
        >>> app.postEvent(obj, QEvent(QEvent.User))
        >>> spy.wait()
        True
        >>> print(spy.events())
        [1000]
        """
        count = len(self.__record)
        self.__timer.stop()
        self.__timer.setInterval(timeout)
        self.__timer.start()
        self.__loop.exec_()
        self.__timer.stop()
        return len(self.__record) != count

    def eventFilter(self, reciever, event):
        if reciever is self.__object and event.type() in self.__etypes:
            self.__record.append(event.type())
            if self.__loop.isRunning():
                self.__loop.quit()
        return super().eventFilter(reciever, event)

    def events(self):
        """
        Return a list of all (listened to) event types that occurred.

        Returns
        -------
        events : List[QEvent.Type]
        """
        return list(self.__record)


@contextlib.contextmanager
def excepthook_catch(raise_on_exit=True):
    """
    Override `sys.excepthook` with a custom handler to record unhandled
    exceptions.

    Use this to capture or note exceptions that are raised and
    unhandled within PyQt slots or virtual function overrides.

    Note
    ----
    The exceptions are still dispatched to the original `sys.excepthook`

    Parameters
    ----------
    raise_on_exit : bool
        If True then the (first) exception that was captured will be
        reraised on context exit

    Returns
    -------
    ctx : ContextManager
        A context manager

    Example
    -------
    >>> class Obj(QObject):
    ...     signal = pyqtSignal()
    ...
    >>> o = Obj()
    >>> o.signal.connect(lambda : 1/0)
    >>> with excepthook_catch(raise_on_exit=False) as exc_list:
    ...    o.signal.emit()
    ...
    >>> print(exc_list)  # doctest: +ELLIPSIS
    [(<class 'ZeroDivisionError'>, ZeroDivisionError('division by zero',), ...
    """
    excepthook = sys.excepthook
    if excepthook != sys.__excepthook__:
        warnings.warn(
            "sys.excepthook was already patched (is {})"
            "(just thought you should know this)".format(excepthook),
            RuntimeWarning, stacklevel=2)
    seen = []

    def excepthook_handle(exctype, value, traceback):
        seen.append((exctype, value, traceback))
        excepthook(exctype, value, traceback)

    sys.excepthook = excepthook_handle
    shouldraise = raise_on_exit
    try:
        yield seen
    except BaseException:
        # propagate/preserve exceptions from within the ctx
        shouldraise = False
        raise
    finally:
        if sys.excepthook == excepthook_handle:
            sys.excepthook = excepthook
        else:
            raise RuntimeError(
                "The sys.excepthook that was installed by "
                "'excepthook_catch' context at enter is not "
                "the one present at exit.")
        if shouldraise and seen:
            raise seen[0][1]


class simulate:
    """
    Utility functions for simulating user interactions with Qt widgets.
    """
    @staticmethod
    def combobox_run_through_all(cbox, delay=-1):
        """
        Run through all items in a given combo box, simulating the user
        focusing the combo box and pressing the Down arrow key activating
        all the items on the way.

        Unhandled exceptions from invoked PyQt slots/virtual function overrides
        are captured and reraised.

        Parameters
        ----------
        cbox : QComboBox
        delay : int
            Run the event loop after the simulated key press (-1, the default,
            means no delay)

        See Also
        --------
        QTest.keyClick
        """
        assert cbox.focusPolicy() & Qt.TabFocus
        cbox.setFocus(Qt.TabFocusReason)
        cbox.setCurrentIndex(-1)
        for i in range(cbox.count()):
            with excepthook_catch() as exlist:
                QTest.keyClick(cbox, Qt.Key_Down, delay=delay)
            if exlist:
                raise exlist[0][1] from exlist[0][1]

    @staticmethod
    def combobox_activate_index(cbox, index, delay=-1):
        """
        Activate an item at `index` in a given combo box.

        The item at index **must** be enabled and selectable.

        Parameters
        ----------
        cbox : QComboBox
        index : int
        delay : int
            Run the event loop after the signals are emitted for `delay`
            milliseconds (-1, the default, means no delay).
        """
        assert 0 <= index < cbox.count()
        model = cbox.model()
        column = cbox.modelColumn()
        root = cbox.rootModelIndex()
        mindex = model.index(index, column, root)
        assert mindex.flags() & Qt.ItemIsEnabled
        cbox.setCurrentIndex(index)
        text = cbox.currentText()
        # QComboBox does not have an interface which would allow selecting
        # the current item as if a user would. Only setCurrentIndex which
        # does not emit the activated signals.
        cbox.activated[int].emit(index)
        cbox.activated[str].emit(text)
        if delay >= 0:
            QTest.qWait(delay)

    @staticmethod
    def combobox_index_of(cbox, value, role=Qt.DisplayRole):
        """
        Find the index of an **selectable** item in a combo box whose `role`
        data contains the given `value`.

        Parameters
        ----------
        cbox : QComboBox
        value : Any
        role : Qt.ItemDataRole

        Returns
        -------
        index : int
            An index such that `cbox.itemData(index, role) == value` **and**
            the item is enabled for selection or -1 if such an index could
            not be found.
        """
        model = cbox.model()
        column = cbox.modelColumn()
        root = cbox.rootModelIndex()
        for i in range(model.rowCount(root)):
            index = model.index(i, column, root)
            if index.data(role) == value and \
                    index.flags() & Qt.ItemIsEnabled:
                pos = i
                break
        else:
            pos = -1
        return pos

    @staticmethod
    def combobox_activate_item(cbox, value, role=Qt.DisplayRole, delay=-1):
        """
        Find an **selectable** item in a combo box whose `role` data
        contains the given value and activate it.

        Raise an ValueError if the item could not be found.

        Parameters
        ----------
        cbox : QComboBox
        value : Any
        role : Qt.ItemDataRole
        delay : int
            Run the event loop after the signals are emitted for `delay`
            milliseconds (-1, the default, means no delay).
        """
        index = simulate.combobox_index_of(cbox, value, role)
        if index < 0:
            raise ValueError("{!r} not in {}".format(value, cbox))
        simulate.combobox_activate_index(cbox, index, delay)


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
