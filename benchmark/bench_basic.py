from .base import Benchmark, benchmark, pandas_only, non_pandas_only
from Orange.data import Table
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq

# noinspection PyBroadException
try:
    from Orange.data.filter import FilterContinuous, FilterDiscrete, Values
except:
    # legacy only
    pass


# noinspection PyStatementEffect
class BenchBasic(Benchmark):
    def setUp(self):
        self.iris = Table('iris')
        self.adult = Table('adult')
        self.discretizer = Discretize(EqualFreq(n=3))

    @benchmark(number=100)
    def bench_iris_read(self):
        Table('iris')

    @benchmark(number=5, warmup=1)
    def bench_adult_read(self):
        Table('adult')

    @benchmark(number=100)
    def bench_iris_create_X(self):
        self.iris.X

    @benchmark(number=50)
    def bench_adult_create_X(self):
        self.adult.X

    @pandas_only
    @benchmark(number=20)
    def bench_adult_filter_pandas(self):
        self.adult[(self.adult.age > 30) & (self.adult.workclass == 'Private')]

    @non_pandas_only
    @benchmark(number=20)
    def bench_adult_filter_pre_pandas(self):
        age_filter = FilterContinuous(self.adult.domain["age"], FilterContinuous.Greater, 30)
        workclass_filter = FilterDiscrete(self.adult.domain["workclass"], [0])
        combined = Values([age_filter, workclass_filter])
        combined(self.adult)

    @benchmark(number=50)
    def bench_iris_basic_stats(self):
        self.iris._compute_basic_stats()

    @benchmark(number=20)
    def bench_iris_distributions(self):
        self.iris._compute_distributions()

    @benchmark()
    def bench_iris_contingency(self):
        self.iris._compute_contingency()

    @benchmark()
    def bench_iris_discretize(self):
        self.discretizer(self.iris)

    @pandas_only
    @benchmark()
    def bench_iris_iteration_pandas(self):
        for _, _ in self.iris.iterrows():
            pass

    @non_pandas_only
    @benchmark()
    def bench_iris_iteration_pre_pandas(self):
        for _ in self.iris:
            pass
