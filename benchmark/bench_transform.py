from functools import partial

import numpy as np
import scipy.sparse
import dask.array

from Orange.data import Table, ContinuousVariable, Domain
from Orange.tests.test_dasktable import temp_dasktable
from Orange.tests.test_table import preprocess_domain_single, preprocess_domain_shared

from .base import Benchmark, benchmark


def add_unknown_attribute(table):
    new_domain = Domain(list(table.domain.attributes) + [ContinuousVariable("x")])
    return table.transform(new_domain)


def add_unknown_class(table):
    new_domain = Domain(table.domain.attributes, class_vars=[ContinuousVariable("x")])
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

    def setup_dask(self, rows, cols):
        table = Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i)) for i in range(cols)]),
            np.random.RandomState(0).rand(rows, cols))
        self.table = temp_dasktable(table)

    @benchmark(setup=partial(setup_dense, rows=10000, cols=100), number=5)
    def bench_copy_dense_long(self):
        add_unknown_attribute(self.table)

    @benchmark(setup=partial(setup_dense, rows=1000, cols=1000), number=5)
    def bench_copy_dense_square(self):
        add_unknown_attribute(self.table)

    @benchmark(setup=partial(setup_dense, rows=100, cols=10000), number=2)
    def bench_copy_dense_wide(self):
        add_unknown_attribute(self.table)

    @benchmark(setup=partial(setup_dask, rows=10000, cols=100), number=5)
    def bench_copy_dask_long(self):
        t = add_unknown_attribute(self.table)
        self.assertIsInstance(t.X, dask.array.Array)

    @benchmark(setup=partial(setup_dask, rows=1000, cols=1000), number=5)
    def bench_copy_dask_square(self):
        t = add_unknown_attribute(self.table)
        self.assertIsInstance(t.X, dask.array.Array)

    @benchmark(setup=partial(setup_dask, rows=100, cols=10000), number=2)
    def bench_copy_dask_wide(self):
        t = add_unknown_attribute(self.table)
        self.assertIsInstance(t.X, dask.array.Array)

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

    @benchmark(setup=partial(setup_dense, rows=10000, cols=100), number=5)
    def bench_subarray_dense_long(self):
        # adding a class should link X
        add_unknown_class(self.table)
        self.assertIs(self.table.X, t.X)

    @benchmark(setup=partial(setup_dask, rows=10000, cols=100), number=5)
    def bench_subarray_dense_long(self):
        # adding a class should link X
        t = add_unknown_class(self.table)
        self.assertIs(self.table.X, t.X)

    def setup_dense_transforms(self, rows, cols, transforms):
        self.setup_dense(rows, cols)
        self.domains = []  # pylint: disable=attribute-defined-outside-init
        self.callbacks = []  # pylint: disable=attribute-defined-outside-init
        domain = self.table.domain
        for t in transforms:
            if t == "single":
                call_cv = None
                domain = preprocess_domain_single(domain, call_cv)
                self.callbacks.append((call_cv,))
            elif t == "shared":
                call_cv, call_shared = None, None
                domain = preprocess_domain_shared(domain, call_cv, call_shared)
                self.callbacks.append((call_cv, call_shared))
            else:
                raise RuntimeError
            self.domains.append(domain)

    @benchmark(setup=partial(setup_dense_transforms, rows=1000, cols=100,
                             transforms=["single"]), number=5)
    def bench_transform_single(self):
        t = self.table.transform(self.domains[-1])
        np.testing.assert_almost_equal(t.X, self.table.X * 2)

    @benchmark(setup=partial(setup_dense_transforms, rows=1000, cols=100,
                             transforms=["single", "single"]), number=5)
    def bench_transform_single_single(self):
        t = self.table.transform(self.domains[-1])
        np.testing.assert_almost_equal(t.X, self.table.X * 2**2)

    @benchmark(setup=partial(setup_dense_transforms, rows=1000, cols=100,
                             transforms=["shared"]), number=5)
    def bench_transform_shared(self):
        t = self.table.transform(self.domains[-1])
        np.testing.assert_almost_equal(t.X, self.table.X * 2)

    @benchmark(setup=partial(setup_dense_transforms, rows=1000, cols=100,
                             transforms=["single", "single", "shared", "single"]), number=5)
    def bench_transform_single_single_shared_single(self):
        t = self.table.transform(self.domains[-1])
        np.testing.assert_almost_equal(t.X, self.table.X * 2**4)


    @benchmark(setup=partial(setup_dense_transforms, rows=1000, cols=100,
                             transforms=["single", "single", "shared",
                                         "single", "shared", "single"]), number=5)
    def bench_transform_single_single_shared_single_shared_single(self):
        t = self.table.transform(self.domains[-1])
        np.testing.assert_almost_equal(t.X, self.table.X * 2**6)
