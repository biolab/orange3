from functools import partial

import numpy as np
import scipy.sparse

from Orange.data import Table, ContinuousVariable, Domain
from .base import Benchmark, benchmark


def add_unknown_attribute(table):
    new_domain = Domain(list(table.domain.attributes) + [ContinuousVariable("x")])
    return table.transform(new_domain)


class BenchTransform(Benchmark):

    def setup_dense(self, rows, cols):
        self.table = Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i)) for i in range(cols)]),
            np.random.RandomState(0).rand(rows, cols))

    def setup_sparse(self, rows, cols):
        sparse = scipy.sparse.rand(rows, cols, density=0.01, format='csr', random_state=0)
        self.table = Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i), sparse=True) for i in range(cols)]),
            sparse)

    @benchmark(setup=partial(setup_dense, rows=10000, cols=100), number=5)
    def bench_copy_dense_long(self):
        add_unknown_attribute(self.table)

    @benchmark(setup=partial(setup_dense, rows=1000, cols=1000), number=5)
    def bench_copy_dense_square(self):
        add_unknown_attribute(self.table)

    @benchmark(setup=partial(setup_dense, rows=100, cols=10000), number=2)
    def bench_copy_dense_wide(self):
        add_unknown_attribute(self.table)

    @benchmark(setup=partial(setup_sparse, rows=10000, cols=100), number=5)
    def bench_copy_sparse_long(self):
        t = add_unknown_attribute(self.table)
        self.assertIsInstance(t.X, scipy.sparse.csr_matrix)

    @benchmark(setup=partial(setup_sparse, rows=1000, cols=1000), number=5)
    def bench_copy_sparse_square(self):
        t = add_unknown_attribute(self.table)
        self.assertIsInstance(t.X, scipy.sparse.csr_matrix)

    @benchmark(setup=partial(setup_sparse, rows=100, cols=10000), number=2)
    def bench_copy_sparse_wide(self):
        t = add_unknown_attribute(self.table)
        self.assertIsInstance(t.X, scipy.sparse.csr_matrix)
