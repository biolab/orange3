import numpy as np

from Orange.data import Table, ContinuousVariable, Domain
from Orange.preprocess import Normalize
from Orange.tests.test_dasktable import temp_dasktable

from .base import Benchmark, benchmark


class BenchNormalize(Benchmark):

    def setUp(self):
        rows = 10000
        cols = 1000
        self.table = Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i)) for i in range(cols)]),
            np.random.RandomState(0).rand(rows, cols))
        self.dasktable = temp_dasktable(self.table)
        self.normalized_domain = Normalize()(self.table).domain
        self.normalized_dasktable = self.dasktable.transform(self.normalized_domain)

    @benchmark(number=5)
    def bench_normalize_dense(self):
        Normalize()(self.table)

    @benchmark(number=5)
    def bench_normalize_dask(self):
        Normalize()(self.dasktable)

    @benchmark(number=5)
    def bench_transform_dense(self):
        self.table.transform(self.normalized_domain)

    @benchmark(number=5)
    def bench_transform_dask(self):
        self.dasktable.transform(self.normalized_domain)

    @benchmark(number=5)
    def bench_transform_dask_values(self):
        self.normalized_dasktable.X.compute()
