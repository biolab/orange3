import sys
import logging

from AnyQt.QtWidgets import QApplication


class WidgetTestRunMixin:
    """A mixin for test-running the widget from a script."""

    @classmethod
    def test_run(cls, input_data=None, **kwargs):
        """
        Create a widget, send it signals, show it, execute it, tear it down.

        This function should be called from the widget module if
        :obj:`__name__` is `__main__` to simplify testing and debugging.

        The function accepts a single argument, which is used as the data for
        the default input signal of the matching type and/or named arguments
        with data for specific input signals.

        Multiple signals can be specified as keyword arguments. A more complex
        sequence of signals can be sent by defining the :obj:`test_run_signal`
        method.

        The function does not return: it ends by calling :obj:`sys.exit` with
        the exit code from the application's main loop.

        Before exiting, the function deletes the reference to the widget
        (after :obj:`test_run_tear_down` already destroyed the widget by using
        :obj:`sip.delete`) and calls Python's garbage collector, as an
        effort to catch any crashes due to widget members (typically
        :obj:`QGraphicsScene` elements) outliving the widget.

        Args:
            input_data: data used for the default input signal of matching type
            **kwargs: data for input signals
        """
        import gc

        widget = cls.test_run_new()
        # pylint: disable=protected-access
        widget._test_run_signals_from_args(input_data, **kwargs)
        widget.test_run_signals()
        exit_code = widget.test_run_exec()
        widget.test_run_tear_down()

        # This can't be done in test_run_tear_down: we want to delete
        # all references to the widget
        del widget
        gc.collect()
        app.processEvents()
        sys.exit(exit_code)

    @classmethod
    def test_run_new(cls):
        """
        Initialize :obj:`QApplication`, and construct and return a new widget.
        """
        global app  # pylint: disable=global-variable-undefined
        app = QApplication(sys.argv)
        logging.basicConfig()
        return cls()

    def test_run_signals(self):
        """
        A method that is called after sending the signals given as arguments
        to `test_run`. Override to send signals for multiple input or to debug
        the behaviour of complicated sequences of signals.
        Don't leave your debug code here, though; move it to unit tests.
        """
        pass

    def _test_run_signals_from_args(self, input_data=None, **kwargs):
        """Send signals given as arguments to :obj:`test_run`"""
        def signal_from_arg():
            if input_data is None:
                return
            inputs = [signal for signal in self.get_signals("inputs")
                      if isinstance(input_data, signal.type)]
            if len(inputs) > 1:
                inputs = [signal for signal in inputs if signal.default]
                if len(inputs) != 1:
                    raise ValueError(
                        "multiple signal handlers for '{}'".
                        format(type(input_data).__name__))
            if not inputs:
                raise ValueError("no signal handlers for '{}'".
                                 format(type(input_data).__name__))
            getattr(self, inputs[0].handler)(input_data)

        def signals_from_kwargs():
            for signal, value in kwargs.items():
                if not isinstance(value, tuple):
                    value = (value,)
                getattr(self, signal)(*value)

        signal_from_arg()
        signals_from_kwargs()

    def test_run_exec(self):
        """Show the widget and start the :obj:`QApplication` main loop."""
        self.handleNewSignals()
        self.show()
        self.raise_()
        return app.exec_()

    def test_run_tear_down(self):
        """Save settings and delete the widget."""
        import sip
        self.saveSettings()
        self.onDeleteWidget()
        sip.delete(self)  #: pylint: disable=c-extension-no-member
