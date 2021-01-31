# Watch what you import in this file,
# it may hang kernel on startup
from collections import defaultdict

from ipykernel.ipkernel import IPythonKernel
from Orange.widgets.data.utils.python_serialize import OrangeZMQMixin

# Sometimes the comm's msg argument isn't used
# pylint: disable=unused-argument


class OrangeIPythonKernel(OrangeZMQMixin, IPythonKernel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables = {}
        self.init_comms_kernel()

    def handle_new_vars(self, vars):
        input_vars = update_kernel_vars(self, vars, self.signals)
        self.variables.update(input_vars)

    async def execute_request(self, *args, **kwargs):
        await super().execute_request(*args, **kwargs)
        if not self.is_initialized():
            return

        variables = collect_kernel_vars(self, self.signals)
        prepared_variables = {
            k[4:] + 's': [v]
            for k, v in variables.items()
        }
        self.set_vars(prepared_variables)


def update_kernel_vars(kernel, vars, signals):
    input_vars = {}

    for signal in signals:
        # remove old out_ vars
        out_name = 'out_' + signal
        if out_name in kernel.shell.user_ns:
            del kernel.shell.user_ns[out_name]
            kernel.shell.user_ns_hidden.pop(out_name, None)

        if signal + 's' in vars and vars[signal + 's']:
            input_vars['in_' + signal + 's'] = vars[signal + 's']

            # prepend script to set single signal values,
            # e.g. in_data = in_datas[0]
            input_vars['in_' + signal] = input_vars['in_' + signal + 's'][0]
        else:
            input_vars['in_' + signal] = None
            input_vars['in_' + signal + 's'] = []
    kernel.shell.push(input_vars)
    return input_vars


def collect_kernel_vars(kernel, signals):
    variables = {}
    for signal in signals:
        name = 'out_' + signal
        if name in kernel.shell.user_ns:
            var = kernel.shell.user_ns[name]
            variables[name] = var
    return variables
