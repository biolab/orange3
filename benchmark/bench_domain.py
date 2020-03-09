import numpy as np

from Orange.data import DomainConversion, Domain, Table, \
    ContinuousVariable, DiscreteVariable
from Orange.preprocess import Discretize, EqualFreq, Normalize

from .base import Benchmark, benchmark


class BenchConversion(Benchmark):

    def setUp(self):
        cols = 1000
        rows = 100
        cont = [ContinuousVariable(str(i)) for i in range(cols)]
        disc = [DiscreteVariable("D" + str(i), values=("1", "2")) for i in range(cols)]
        self.domain = Domain(cont + disc)
        self.domain_x = Domain(list(self.domain.attributes) + [ContinuousVariable("x")])
        self.single = Domain([ContinuousVariable("0")])
        self.table = Table.from_numpy(
            self.domain,
            np.random.RandomState(0).randint(0, 2, (rows, len(self.domain))))
        self.discretized_domain = Discretize(EqualFreq(n=3))(self.table).domain
        self.normalized_domain = Normalize()(self.table).domain

    @benchmark(number=5)
    def bench_full(self):
        DomainConversion(self.domain, self.domain_x)

    @benchmark(number=5)
    def bench_selection(self):
        DomainConversion(self.domain, self.single)

    @benchmark(number=5)
    def bench_transform_copy(self):
        self.table.transform(self.domain_x)

    @benchmark(number=5)
    def bench_transform_discretize(self):
        self.table.transform(self.discretized_domain)

    @benchmark(number=5)
    def bench_transform_normalize(self):
        self.table.transform(self.normalized_domain)
