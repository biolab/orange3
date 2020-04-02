from Orange.data import ContinuousVariable
from .base import Benchmark, benchmark


class BenchContinuous(Benchmark):

    # pylint: disable=no-self-use
    @benchmark()
    def bench_str_val_decimals(self):
        a = ContinuousVariable("a", 4)
        for _ in range(1000):
            a.str_val(1.23456)

    # pylint: disable=no-self-use
    @benchmark()
    def bench_str_val_g(self):
        a = ContinuousVariable("a")
        for _ in range(1000):
            a.str_val(1.23456)
