import sys
import logging
import gc

from AnyQt.QtWidgets import QApplication


class WidgetPreview:
    """
    A helper class for widget previews.

    Attributes:
        widget (OWWidget): an instance of the widget or `None`
        widget_cls (type): the widget class
    """

    def __init__(self, widget_cls):
        self.widget_cls = widget_cls
        self.widget = None
        logging.basicConfig()

    def run(self, input_data=None, *, no_exec=False, no_exit=False, **kwargs):
        """
        Run a preview of the widget;

        It first creates a widget, unless it exists from the previous call.
        This can only happen if `no_exit` was set to `True`.

        Next, it passes the data signals to the widget. Data given as
        positional argument must be of a type for which there exist a single
        or a default handler. Signals can also be given by keyword arguments,
        where the name of the argument is the name of the handler method.
        If the data is a list of tuples, the sequence of tuples is sent to
        the same handler.

        Next, the method shows the widget and starts the event loop, unless
        `no_exec` argument is set to `True`.

        Finally, unless the argument `no_exit` is set to `True`, the method
        tears down the widget, deletes the reference to the widget and calls
        Python's garbage collector, as an effort to catch any crashes due to
        widget members (typically :obj:`QGraphicsScene` elements) outliving
        the widget. It then calls :obj:`sys.exit` with the exit code from
        the application's main loop.

        If `no_exit` is set to `True`, the `run` keeps the widget alive.
        In this case, subsequent calls to `run` or other methods
        (`send_signals`, `exec_widget`) will use the same widget.

        Args:
            input_data: data used for the default input signal of matching type
            no_exec (bool): if set to `True`, the widget is not shown and the
                event loop is not started
            no_exit (bool): if set to `True`, the widget is not torn down
            **kwargs: data for input signals
        """
        if self.widget is None:
            self.create_widget()
        self.send_signals(input_data, **kwargs)
        if not no_exec:
            exit_code = self.exec_widget()
        else:
            exit_code = 0
        if not no_exit:
            self.tear_down()
            sys.exit(exit_code)

    def create_widget(self):
        """
        Initialize :obj:`QApplication` and construct the widget.
        """
        global app  # pylint: disable=global-variable-undefined
        app = QApplication(sys.argv)
        self.widget = self.widget_cls()

    def send_signals(self, input_data=None, **kwargs):
        """Send signals to the widget"""

        def call_handler(handler_name, data):
            handler = getattr(self.widget, handler_name)
            for chunk in self._data_chunks(data):
                handler(*chunk)

        if input_data is not None:
            handler_name = self._find_handler_name(input_data)
            call_handler(handler_name, input_data)
        for handler_name, data in kwargs.items():
            call_handler(handler_name, data)
        self.widget.handleNewSignals()

    def _find_handler_name(self, data):
        chunk = next(self._data_chunks(data))[0]
        chunk_type = type(chunk).__name__
        inputs = [signal
                  for signal in self.widget.get_signals("inputs")
                  if isinstance(chunk, signal.type)]
        if not inputs:
            raise ValueError(f"no signal handlers for '{chunk_type}'")
        if len(inputs) > 1:
            inputs = [signal for signal in inputs if signal.default]
            if len(inputs) != 1:
                raise ValueError(
                    f"multiple signal handlers for '{chunk_type}'")
        return inputs[0].handler

    @staticmethod
    def _data_chunks(data):
        if isinstance(data, list) \
                and data \
                and all(isinstance(x, tuple) for x in data):
            yield from iter(data)
        elif isinstance(data, tuple):
            yield data
        else:
            yield (data,)

    def exec_widget(self):
        """Show the widget and start the :obj:`QApplication`'s main loop."""
        self.widget.show()
        self.widget.raise_()
        return app.exec_()

    def tear_down(self):
        """Save settings and delete the widget."""
        import sip
        self.widget.saveSettings()
        self.widget.onDeleteWidget()
        sip.delete(self.widget)  #: pylint: disable=c-extension-no-member
        self.widget = None
        gc.collect()
        app.processEvents()
