import codecs
import pickle
from collections import defaultdict

from ipykernel.ipkernel import IPythonKernel

from Orange.widgets.data.owpythonscript import OWPythonScript


class OrangeIPythonKernel(IPythonKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.comm_variables = defaultdict(list)

        self.comm_manager.register_target('run_script', self.run_script)

    def run_script(self, comm, open_msg):
        comm.on_msg(
            lambda msg: self._on_run_script_request(comm, msg)
        )

    def _on_run_script_request(self, comm, msg):
        data = msg['content']['data']
        script = data['script']
        inputs = data['locals']

        variables = {
            k: pickle.loads(codecs.decode(l.encode(), 'base64'))
            for k, l in inputs.items()
        }

        self.shell.push(variables)
        self.comm_variables[comm] = list(variables)

        future = self.do_execute(script, silent=False)
        future.add_done_callback(
            lambda f: self._on_run_script_done(comm, f)
        )

    def _on_run_script_done(self, comm, future):
        try:
            result = future.done()
        # exception stack traces are shown in console anyway
        except Exception:
            return

        outputs = {}
        for signal in OWPythonScript.signal_names:
            name = 'out_' + signal
            ns = self.shell.user_ns
            out = OWPythonScript.Outputs.__dict__[signal]
            if name in ns and isinstance(ns[name], out.type):
                outputs[name] = ns[name]

        comm.send({
            k: codecs.encode(pickle.dumps(o), 'base64').decode()
            for k, o in outputs.items()
        })


