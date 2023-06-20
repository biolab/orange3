from unittest.mock import patch, MagicMock

import numpy as np

from Orange.data import Domain, Table, ContinuousVariable
from Orange.preprocess import Normalize, SklImpute

from .base import Benchmark, benchmark


class SetUpData:

    def setUp(self):
        cols = 1000
        rows = 1000
        cont = [ContinuousVariable(str(i)) for i in range(cols)]
        self.domain = Domain(cont)
        self.single = Domain([ContinuousVariable("0")])
        self.table = Table.from_numpy(
            self.domain,
            np.random.RandomState(0).randint(0, 2, (rows, len(self.domain.variables))))
        self.normalized_domain = Normalize()(self.table).domain


class BenchNormalize(SetUpData, Benchmark):

    @benchmark(number=5)
    def bench_normalize_only_transform(self):
        self.table.transform(self.normalized_domain)

    @benchmark(number=5)
    def bench_normalize_only_parameters(self):
        # avoid benchmarking transformation
        with patch("Orange.data.Table.transform", MagicMock()):
            Normalize()(self.table)


class BenchSklImpute(SetUpData, Benchmark):

    @benchmark(number=5)
    def bench_sklimpute(self):
        SklImpute()(self.table)
