import contextlib
import time
import warnings

from AnyQt.QtCore import QEventLoop, pyqtSignal as Signal, pyqtProperty
from AnyQt.QtWidgets import QApplication, qApp

from Orange.widgets import gui

class ProgressBarMixin:
    # Set these here so we avoid having to call `__init__` fromm classes
    # that use this mix-in
    __progressBarValue = -1
    __progressState = 0
    startTime = time.time()  # used in progressbar

    def progressBarInit(self, processEvents=QEventLoop.AllEvents):
        """
        Initialize the widget's progress (i.e show and set progress to 0%).

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        self.startTime = time.time()
        self.setWindowTitle(self.captionTitle + " (0% complete)")

        if self.__progressState != 1:
            self.__progressState = 1
            self.processingStateChanged.emit(1)

        self.progressBarSet(0, processEvents)

    def progressBarSet(self, value, processEvents=QEventLoop.AllEvents):
        """
        Set the current progress bar to `value`.

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param float value: Progress value
        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        old = self.__progressBarValue
        self.__progressBarValue = value

        if value > 0:
            if self.__progressState != 1:
                warnings.warn("progressBarSet() called without a "
                              "preceding progressBarInit()",
                              stacklevel=2)
                self.__progressState = 1
                self.processingStateChanged.emit(1)

            usedTime = max(1, time.time() - self.startTime)
            totalTime = 100.0 * usedTime / value
            remainingTime = max(0, int(totalTime - usedTime))
            hrs = remainingTime // 3600
            mins = (remainingTime % 3600) // 60
            secs = remainingTime % 60
            if hrs > 0:
                text = "{}:{:02}:{:02}".format(hrs, mins, secs)
            else:
                text = "{}:{}:{:02}".format(hrs, mins, secs)
            self.setWindowTitle("{} ({:d}%, ETA: {})"
                                .format(self.captionTitle, int(value), text))
        else:
            self.setWindowTitle(self.captionTitle + " (0% complete)")

        if old != value:
            self.progressBarValueChanged.emit(value)

        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    def progressBarValue(self):
        """Return the state of the progress bar
        """
        return self.__progressBarValue

    progressBarValue = pyqtProperty(
        float, fset=progressBarSet, fget=progressBarValue)
    processingState = pyqtProperty(int, fget=lambda self: self.__progressState)

    def progressBarAdvance(self, value, processEvents=QEventLoop.AllEvents):
        """
        Advance the progress bar.

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        Args:
            value (int): progress value
            processEvents (`QEventLoop.ProcessEventsFlags` or `None`):
                process events flag
        """
        self.progressBarSet(self.progressBarValue + value, processEvents)

    def progressBarFinished(self, processEvents=QEventLoop.AllEvents):
        """
        Stop the widget's progress (i.e hide the progress bar).

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        self.setWindowTitle(self.captionTitle)
        if self.__progressState != 0:
            self.__progressState = 0
            self.processingStateChanged.emit(0)

        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    @contextlib.contextmanager
    def progressBar(self, iterations=0):
        """
        Context manager for progress bar.

        Using it ensures that the progress bar is removed at the end without
        needing the `finally` blocks.

        Usage:

            with self.progressBar(20) as progress:
                ...
                progress.advance()

        or

            with self.progressBar() as progress:
                ...
                progress.advance(0.15)

        or

            with self.progressBar():
                ...
                self.progressBarSet(50)

        :param iterations: the number of iterations (optional)
        :type iterations: int
        """
        progress_bar = gui.ProgressBar(self, iterations)
        try:
            yield progress_bar
        finally:
            progress_bar.finish()
