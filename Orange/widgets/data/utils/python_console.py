import codecs
import logging

import itertools
import pickle
import threading

from AnyQt.QtCore import Qt, Signal
from Orange.widgets.data.utils.python_serialize import OrangeZMQMixin
from qtconsole.client import QtKernelClient
from qtconsole.rich_jupyter_widget import RichJupyterWidget

# Sometimes the comm's msg argument isn't used
# pylint: disable=unused-argument
# pylint being stupid? in_prompt is defined as a class var in JupyterWidget
# pylint: disable=attribute-defined-outside-init

log = logging.getLogger(__name__)


class OrangeConsoleWidget(OrangeZMQMixin, RichJupyterWidget):
    becomes_ready = Signal()

    execution_started = Signal()

    execution_finished = Signal(bool)  # False for error

    results_ready = Signal(dict)

    begun_collecting_variables = Signal()

    variables_finished_injecting = Signal()

    def __init__(self, *args, style_sheet='', **kwargs):
        super().__init__(*args, **kwargs)
        self.__is_ready = False

        self.__queued_broadcast = None
        self.__queued_execution = None
        self.__prompt_num = 1
        self.__default_in_prompt = self.in_prompt
        self.__executing = False
        self.__broadcasting = False
        self.__threads = []

        self.inject_vars_comm = None
        self.collect_vars_comm = None

        self.style_sheet = style_sheet + \
                           '.run-prompt { color: #aa22ff; }'

        # Let the widget/kernel start up before trying to run a script,
        # by storing a queued execution payload when the widget's commit
        # method is invoked before <In [0]:> appears.
        @self.becomes_ready.connect
        def _():
            self.becomes_ready.disconnect(_)  # reset callback
            self.init_client()
            self.becomes_ready.connect(self.__on_ready)
            self.__on_ready()

    def __on_ready(self):
        self.__is_ready = True
        self.__run_queued_broadcast()
        self.__run_queued_payload()

    def __run_queued_broadcast(self):
        if not self.__is_ready or self.__queued_broadcast is None:
            return
        qb = self.__queued_broadcast
        self.__queued_broadcast = None
        self.set_vars(*qb)

    def __run_queued_payload(self):
        if not self.__is_ready or self.__queued_execution is None:
            return
        qe = self.__queued_execution
        self.__queued_execution = None
        self.run_script(*qe)

    def run_script(self, script):
        """
        Inject the in vars, run the script,
        collect the out vars (emit the results_ready signal).
        """
        if not self.__is_ready:
            self.__queued_execution = (script, )
            return

        if self.__executing or self.__broadcasting:
            self.__queued_execution = (script, )
            self.__is_ready = False
            if self.__executing:
                self.interrupt_kernel()
            return

        # run the script
        self.__executing = True
        log.debug('Running script')
        # update prompts
        self._set_input_buffer('')
        self.in_prompt = '<span class="run-prompt">' \
                         'Run[<span class="in-prompt-number">%i</span>]' \
                         '</span>'
        self._update_prompt(self.__prompt_num)
        self._append_plain_text('\n')
        self.in_prompt = 'Running script...'
        self._show_interpreter_prompt(self.__prompt_num)

        self.execution_started.emit()
        # we abuse this method instead of others to keep
        # the 'Running script...' prompt at the bottom of the console
        self.kernel_client.execute(script)

    def set_vars(self, vars):
        if not self.__is_ready:
            self.__queued_broadcast = (vars, )
            return

        if self.__executing or self.__broadcasting:
            self.__is_ready = False
            self.__queued_broadcast = (vars, )
            return

        self.__broadcasting = True

        self.in_prompt = "Injecting variables..."
        self._update_prompt(self.__prompt_num)

        super().set_vars(vars)

    def on_variables_injected(self):
        log.debug('Cleared injecting variables')
        self.__broadcasting = False
        self.in_prompt = self.__default_in_prompt
        self._update_prompt(self.__prompt_num)

        self.variables_finished_injecting.emit()

        if not self.__is_ready:
            self.becomes_ready.emit()

    def on_start_collecting_vars(self):
        log.debug('Collecting variables...')

        # the prompt isn't updated to reflect this,
        # but the widget should show that variables are being collected

        # self.in_prompt = 'Collecting variables...'
        # self._update_prompt(self.__prompt_num)
        self.begun_collecting_variables.emit()

    def handle_new_vars(self, vardict):
        varlists = {
            'out_' + name[:-1]: vs[0]
            for name, vs in vardict.items()
            if len(vs) > 0
        }

        self.results_ready.emit(varlists)

    # override
    def _handle_execute_result(self, msg):
        super()._handle_execute_result(msg)
        if self.__executing:
            self._append_plain_text('\n', before_prompt=True)

    # override
    def _handle_execute_reply(self, msg):
        if 'execution_count' in msg['content']:
            self.__prompt_num = msg['content']['execution_count'] + 1

        if not self.__executing:
            super()._handle_execute_reply(msg)
            return

        self.__executing = False
        self.in_prompt = self.__default_in_prompt

        if msg['content']['status'] != 'ok':
            self.execution_finished.emit(False)
            self._show_interpreter_prompt(self.__prompt_num)
            super()._handle_execute_reply(msg)
            return

        self._update_prompt(self.__prompt_num)
        self.execution_finished.emit(True)

    # override
    def _handle_kernel_died(self, since_last_heartbeat):
        super()._handle_kernel_died(since_last_heartbeat)
        self.__is_ready = False

    # override
    def _show_interpreter_prompt(self, number=None):
        """
        The console's ready when the prompt shows up.
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
