import codecs
import pickle
import sys
from collections import defaultdict

from ipykernel.ipkernel import IPythonKernel

from Orange.widgets.data.owpythonscript import OWPythonScript


class OrangeIPythonKernel(IPythonKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm_variables = defaultdict(list)

        self.comm_manager.register_target('inject_vars', self.inject_vars)
        self.comm_manager.register_target('collect_vars', self.collect_vars)

    def inject_vars(self, comm, open_msg):
        comm.on_msg(
            lambda msg: self._on_inject_vars_request(comm, msg)
        )

    def _on_inject_vars_request(self, comm, msg):
        data = msg['content']['data']
        inputs = data['locals']

        input_vars = {
            k: pickle.loads(codecs.decode(l.encode(), 'base64'))
            for k, l in inputs.items()
        }

        # remove old in_ vars
        for v in self.comm_variables[comm]:
            if v not in input_vars:
                del self.shell.user_ns[v]
                self.shell.user_ns_hidden.pop(v, None)
        # remove old out_ vars
        for signal in OWPythonScript.signal_names:
            name = 'out_' + signal
            if name in self.shell.user_ns:
                del self.shell.user_ns[name]
                self.shell.user_ns_hidden.pop(name, None)

        self.shell.push(input_vars)
        self.comm_variables[comm] = list(input_vars)

        comm.send({
            'status': 'ok'
        })

    def collect_vars(self, comm, open_msg):
        comm.on_msg(
            lambda msg: self._on_collect_vars_request(comm, msg)
        )

    def _on_collect_vars_request(self, comm, msg):
        outputs = {}
        for signal in OWPythonScript.signal_names:
            name = 'out_' + signal
            ns = self.shell.user_ns
            out = OWPythonScript.Outputs.__dict__[signal]
            if name in ns:
                outputs[name] = ns[name]

        serialized_vars = {
            k: codecs.encode(pickle.dumps(o), 'base64').decode()
            for k, o in outputs.items()
        }
        comm.send({'outputs': serialized_vars})
