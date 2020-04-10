from functools import partial
import os

import numpy as np
import scipy.sparse

from Orange.data import Table, ContinuousVariable, Domain
from .base import Benchmark, benchmark


def save(table, fn):
    try:
        table.save(fn)
    finally:
        os.remove(fn)


class BenchSave(Benchmark):

    def setup_dense(self, rows, cols, varkwargs=None):
        if varkwargs is None:
            varkwargs = {}
        self.table = Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i), **varkwargs) for i in range(cols)]),
            np.random.RandomState(0).rand(rows, cols))

    def setup_sparse(self, rows, cols, varkwargs=None):
        if varkwargs is None:
            varkwargs = {}
        sparse = scipy.sparse.rand(rows, cols, density=0.01, format='csr', random_state=0)
        self.table = Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i), sparse=True, **varkwargs) for i in range(cols)]),
            sparse)

    @benchmark(setup=partial(setup_dense, rows=100, cols=10))
    def bench_print_dense(self):
        str(self.table)

    @benchmark(setup=partial(setup_dense, rows=100, cols=10,
                             varkwargs={"number_of_decimals": 2}))
    def bench_print_dense_decimals(self):
        str(self.table)

    @benchmark(setup=partial(setup_sparse, rows=100, cols=10), number=5)
    def bench_print_sparse(self):
        str(self.table)

    @benchmark(setup=partial(setup_sparse, rows=100, cols=10,
                             varkwargs={"number_of_decimals": 2}),
               number=5)
    def bench_print_sparse_decimals(self):
        str(self.table)

    @benchmark(setup=partial(setup_dense, rows=100, cols=100))
    def bench_save_tab(self):
        save(self.table, "temp_save.tab")

    @benchmark(setup=partial(setup_dense, rows=100, cols=100,
                             varkwargs={"number_of_decimals": 2}))
    def bench_save_tab_decimals(self):
        save(self.table, "temp_save.tab")
