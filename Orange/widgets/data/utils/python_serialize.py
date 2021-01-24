from _weakref import ref

import logging
import pickle
import zlib
from collections import defaultdict

import threading

from weakref import WeakValueDictionary, WeakKeyDictionary

import numpy
import zmq

# avoid import Qt here, it'll slow down kernel startup considerably

log = logging.getLogger(__name__)


class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods
    send_zipped_pickle is just like send_pyobj, but uses
    zlib to compress the stream before sending.
    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    """

    def send_zipped_pickle(self, obj, flags=0, protocol=-1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        log.info('zipped pickle is %i bytes' % len(zobj))
        return self.send(zobj, flags=flags)

    def recv_zipped_pickle(self, flags=0):
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

    def send_array(self, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = {
            'dtype': str(A.dtype),
            'shape': A.shape,
        }
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)  # TYL
        A = numpy.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])

    # def send_table(self, T, flags=0, copy=True, track=False):
    #
    # def recv_table(self, flags=0, copy=True, track=False):
    #     kwargs = {
    #         arrname: self.recv_array(comm, flags, copy, track)
    #         for arrname, comm in table['arraycomms'].items()
    #     }
    #     kwargs['domain'] = deserialize_object(table['domain'])
    #     kwargs['attributes'] = deserialize_object(table['attributes'])
    #     from Orange.data import Table
    #     return Table.from_numpy(**kwargs)

    def send_vars(self, variables, flags=0, copy=True, track=False):
        vars = defaultdict(list)
        vars.update(variables)

        tables = vars['datas']
        models = vars['classifiers']
        learners = vars['learners']
        objects = vars['objects']

        # all of this is sent only once the non-multipart send_string initiates

        # introduce receiver to vars
        self.send_json(
            {
                vn: [v[0] for v in vs]
                for vn, vs in vars.items()
            },
            flags=flags | zmq.SNDMORE
        )
        # send vars
        for t in [t[1] for t in tables]:
            self.send_zipped_pickle(t)
        for m in [m[1] for m in models]:
            self.send_zipped_pickle(m)
        for l in [l[1] for l in learners]:
            self.send_zipped_pickle(l)
        for o in [o[1] for o in objects]:
            self.send_zipped_pickle(o)
        # initiate multipart msg
        self.send_string('')

    def recv_vars(self, flags=0, copy=True, track=False):
        spec = self.recv_json(flags)
        tables = [
            (i, self.recv_zipped_pickle())
            for i in spec['datas']
        ]
        models = [
            (i, self.recv_zipped_pickle())
            for i in spec['classifiers']
        ]
        learners = [
            (i, self.recv_zipped_pickle())
            for i in spec['learners']
        ]
        objects = [
            (i, self.recv_zipped_pickle())
            for i in spec['objects']
        ]
        self.recv_string()

        return {
            'datas': tables,
            'classifiers': models,
            'learners': learners,
            'objects': objects
        }


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


class OrangeZMQMixin:

    signals = ('data', 'learner', 'classifier', 'object')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__kernel_id = None

        self.__threads = []
        self.__my_vars_for_id = WeakKeyDictionary()
        self.__received_vars_by_id = WeakValueDictionary()
        self.__my_vars = {}  # e.g., {'tables': [(5, table_object)], ...}
        self.__held_vars = {}  # e.g., {'tables': [(5, table_object)], ...}
        self.__last_id = 0

        self.init_comm = None
        self.broadcast_comm = None
        self.request_comm = None

        self.ctx = SerializingContext()
        self.ctx.setsockopt(zmq.RCVTIMEO, 300000)
        self.ctx.setsockopt(zmq.SNDTIMEO, 300000)

        self.socket = self.ctx.socket(zmq.PAIR)

    # Abstract interface

    def handle_new_vars(self, vars):
        raise NotImplementedError

    def on_variables_injected(self):
        pass

    def on_start_collecting_vars(self):
        pass

    # Methods

    def set_kernel_id(self, kernel_id, kernel=False):
        if self.__kernel_id == kernel_id:
            return
        self.__kernel_id = kernel_id

        self.__my_vars_for_id = WeakKeyDictionary()
        self.__received_vars_by_id = WeakValueDictionary()
        self.__my_vars = {}  # e.g., {'tables': [(5, table_object)], ...}
        self.__held_vars = {}  # e.g., {'tables': [(5, table_object)], ...}
        self.socket = self.ctx.socket(zmq.PAIR)

        if kernel:
            self.init_socket_kernel()
        elif self.init_comm is not None:
            self.init_comm.send({
                'id': kernel_id
            })

    def set_vars(self, vars):
        self.__my_vars = self.__identify_vars(vars)
        varspec = {
            name: [i for i, _ in vs]
            for name, vs in self.__my_vars.items()
        }
        self.sync_vars(varspec)

    def sync_vars(self, varspec):
        if self.broadcast_comm is not None:
            self.__broadcast_vars(varspec)

    def is_initialized(self):
        return self.__kernel_id is not None

    def init_socket_kernel(self):
        self.socket.bind('ipc://' + self.__kernel_id)

    def init_comms_kernel(self):
        def comm_init(comm_name, callback):
            def assign_comm(comm, _):
                setattr(self, comm_name, comm)
                comm.on_msg(callback)
            return assign_comm

        self.comm_manager.register_target(
            'request_comm',
            comm_init(
                'request_comm',
                lambda msg: self.__on_comm_request(msg)
            )
        )
        self.comm_manager.register_target(
            'broadcast_comm',
            comm_init(
                'broadcast_comm',
                lambda msg: self.__on_comm_broadcast(msg)
            )
        )
        self.comm_manager.register_target(
            'init_comm',
            comm_init(
                'init_comm',
                lambda msg: self.__on_comm_init(msg)
            )
        )

    def init_client(self):
        self.socket.connect('ipc://' + self.__kernel_id)
        self.request_comm = self.kernel_client.comm_manager.new_comm(
            'request_comm', {}
        )
        self.request_comm.on_msg(self.__on_comm_request)
        self.broadcast_comm = self.kernel_client.comm_manager.new_comm(
            'broadcast_comm', {}
        )
        self.broadcast_comm.on_msg(self.__on_comm_broadcast)
        self.init_comm = self.kernel_client.comm_manager.new_comm(
            'init_comm', {}
        )
        if self.__kernel_id is not None:
            self.init_comm.send({
                'id': self.__kernel_id
            })

    # Private parts

    def __on_comm_broadcast(self, msg):
        varspec = msg['content']['data']['varspec']

        self.__held_vars = {
            name: [
                (i, self.__received_vars_by_id.get(i, None))
                for i in is_
            ]
            for name, is_ in varspec.items()
        }

        missing_ids = []
        for name, vs in self.__held_vars.items():
            for i, var in vs:
                if var is None:
                    missing_ids.append(i)

        if missing_ids:
            self.__recv_vars(
                callback=self.__finalize_vars
            )
            msg = {
                'status': 'missing',
                'var_ids': missing_ids
            }
            self.on_start_collecting_vars()
        else:
            msg = {
                'status': 'ok'
            }
            self.__finalize_vars(self.__held_vars)
        self.request_comm.send(msg)

    def __on_comm_request(self, msg):
        if msg['content']['data']['status'] == 'ok':
            self.on_variables_injected()
        else:
            var_ids = msg['content']['data']['var_ids']
            payload = {
                name: [
                    v for v in vs
                    if v[0] in var_ids
                ]
                for name, vs in self.__my_vars.items()
            }
            self.__send_vars(payload)

    def __on_comm_init(self, msg):
        i = msg['content']['data']['id']
        self.set_kernel_id(i, kernel=True)

    def __identify_vars(self, vars):
        vars_with_ids = defaultdict(list)
        for name, vs in vars.items():
            for var in vs:

                # if the object is not weak referencible,
                # it's going to be copied each time
                try:
                    ref(var)
                except TypeError:
                    new_id = self.__new_id()
                    vars_with_ids[name].append((new_id, var))
                    continue

                i = self.__my_vars_for_id.get(var, None)
                if i is not None:
                    vars_with_ids[name].append((i, var))
                else:
                    new_id = self.__new_id()
                    self.__my_vars_for_id[var] = new_id
                    vars_with_ids[name].append((new_id, var))
        return vars_with_ids

    def __finalize_vars(self, vars):
        self.__held_vars.update(vars)

        var_objs = {
            k: [v[1] for v in vs]
            for k, vs in self.__held_vars.items()
        }

        self.handle_new_vars(var_objs)
        self.request_comm.send({
            'status': 'ok'
        })

    def __send_vars(self, vars, callback=lambda *_: None):
        self.__run_thread_with_callback(
            self.socket.send_vars,
            (vars, ),
            callback
        )

    def __recv_vars(self, callback=lambda *_: None):
        self.__run_thread_with_callback(
            self.socket.recv_vars,
            (),
            callback
        )

    def __broadcast_vars(self, varspec):
        self.broadcast_comm.send({
            'varspec': varspec
        })

    def __run_thread_with_callback(self, target, args, callback):

        def target_and_callback():
            result = target(*args)
            self.__threads.remove(thread)
            if result is not None:
                callback(result)
            else:
                callback()

        thread = threading.Thread(
            target=target_and_callback
        )
        self.__threads.append(thread)
        thread.start()

    def __new_id(self):
        self.__last_id += 1
        return self.__last_id
