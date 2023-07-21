import numpy as np

from Orange.data import Table, ContinuousVariable, Domain
from Orange.preprocess import Normalize, SklImpute, PreprocessorList
from Orange.tests.test_dasktable import temp_dasktable

from .base import Benchmark, benchmark


class BenchNormalize(Benchmark):

    preprocessor = Normalize()

    @classmethod
    def create_data(cls):
        rows = 10000
        cols = 1000
        return Table.from_numpy(  # pylint: disable=W0201
            Domain([ContinuousVariable(str(i)) for i in range(cols)]),
            np.random.RandomState(0).rand(rows, cols))

    @classmethod
    def setUpClass(cls):
        cls.table = cls.create_data()
        cls.dasktable = temp_dasktable(cls.table)
        cls.preprocessed_domain = cls.preprocessor(cls.table).domain
        cls.preprocessed_dasktable = cls.dasktable.transform(cls.preprocessed_domain)

    @benchmark(number=3, warmup=1)
    def bench_run_dense(self):
        self.preprocessor(self.table)

    @benchmark(number=3, warmup=1)
    def bench_run_dask(self):
        self.preprocessor(self.dasktable)

    @benchmark(number=3, warmup=1)
    def bench_transform_dense(self):
        self.table.transform(self.preprocessed_domain)

    @benchmark(number=3, warmup=1)
    def bench_transform_dask(self):
        self.dasktable.transform(self.preprocessed_domain)

    @benchmark(number=3, warmup=1)
    def bench_transform_dask_values(self):
        self.preprocessed_dasktable.X.compute()


class BenchSkImpute(BenchNormalize):
    preprocessor = SklImpute()


class BenchNormalizeImpute(BenchNormalize):
    preprocessor = PreprocessorList([Normalize(), SklImpute()])


class BenchImputeNormalize(BenchNormalize):
    preprocessor = PreprocessorList([SklImpute(), Normalize()])
