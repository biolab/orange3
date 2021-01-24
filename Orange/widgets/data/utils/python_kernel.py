# Watch what you import in this file,
# it may hang kernel on startup
from collections import defaultdict

from ipykernel.ipkernel import IPythonKernel
from Orange.widgets.data.utils.python_serialize import OrangeZMQMixin

# Sometimes the comm's msg argument isn't used
# pylint: disable=unused-argument


class OrangeIPythonKernel(OrangeZMQMixin, IPythonKernel):

    signals = ("data", "learner", "classifier", "object")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables = defaultdict(list)
        self.init_comms_kernel()
        self.handle_new_vars({})

    def handle_new_vars(self, vars):
        default_vars = defaultdict(list)
        default_vars.update(vars)

        input_vars = {}

        for signal in self.signals:
            # remove old out_ vars
            out_name = 'out_' + signal
            if out_name in self.shell.user_ns:
                del self.shell.user_ns[out_name]
                self.shell.user_ns_hidden.pop(out_name, None)

            if signal + 's' in vars and vars[signal + 's']:
                input_vars['in_' + signal + 's'] = vars[signal + 's']

                # prepend script to set single signal values,
                # e.g. in_data = in_datas[0]
                input_vars['in_' + signal] = input_vars['in_' + signal + 's'][0]
            else:
                input_vars['in_' + signal] = None
                input_vars['in_' + signal + 's'] = []

        self.shell.push(input_vars)
        self.variables.update(input_vars)

    async def execute_request(self, *args, **kwargs):
        await super().execute_request(*args, **kwargs)
        if not self.is_initialized():
            return

        vars = defaultdict(list)
        for signal in self.signals:
            key = signal + 's'
            name = 'out_' + signal
            if name in self.shell.user_ns:
                var = self.shell.user_ns[name]
                vars[key].append(var)

        self.set_vars(vars)

        return result
