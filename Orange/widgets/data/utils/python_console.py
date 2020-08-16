import codecs
import pickle

from AnyQt.QtCore import Qt, Signal
from qtconsole.rich_jupyter_widget import RichJupyterWidget

# Sometimes the comm's msg argument isn't used
# pylint: disable=unused-argument
# pylint being stupid? in_prompt is defined as a class var in JupyterWidget
# pylint: disable=attribute-defined-outside-init


class OrangeConsoleWidget(RichJupyterWidget):
    becomes_ready = Signal()

    execution_finished = Signal(bool)  # False for error

    results_ready = Signal(dict)

    def __init__(self, *args, style_sheet='', **kwargs):
        super().__init__(*args, **kwargs)
        self.__queued_execution = None
        self.__prompt_num = 1
        self.__default_in_prompt = self.in_prompt
        self.__executing = False
        self.__is_ready = False

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
            lambda success: self.collect_vars_comm.send({}) if success else None
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

    def run_script_with_locals(self, script, local_vars):
        """
        Inject the in vars, run the script,
        collect the out vars (emit the results_ready signal).
        """
        if not self.__is_ready:
            self.__queued_execution = (script, local_vars)
            return

        if self.__executing:
            if not self.__queued_execution:
                @self.execution_finished.connect
                def _():
                    self.execution_finished.disconnect(_)  # reset callback
                    self.__run_queued_payload()
            self.__queued_execution = (script, local_vars)
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
            for k, l in local_vars.items()
        }})

    def __on_variables_injected(self, msg, script):
        # update prompts
        self._set_input_buffer('')
        self.in_prompt = '<span class="run-prompt">' \
                         'Run[<span class="in-prompt-number">%i</span>]' \
                         '</span>'
        self._update_prompt(self.__prompt_num)
        self._append_plain_text('\n')
        self.in_prompt = 'Running script...'
        self._show_interpreter_prompt(self.__prompt_num)

        # run the script
        self.__executing = True
        # we abuse this method instead of others to keep
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
