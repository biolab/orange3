import numpy as np

from Orange.data import Domain, Table, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owkmeans import OWKMeans

from .base import benchmark


def table(rows, cols):
    return Table.from_numpy(  # pylint: disable=W0201
        Domain([ContinuousVariable(str(i)) for i in range(cols)]),
        np.random.RandomState(0).rand(rows, cols))


class BenchOWKmeans(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.d_100_100 = table(100, 100)
        cls.d_sampled_silhouette = table(10000, 1)
        cls.d_10_500 = table(10, 500)

    def setUp(self):
        self.widget = None  # to avoid lint errors

    def widget_from_to(self):
        self.widget = self.create_widget(
            OWKMeans, stored_settings={"auto_commit": False})
        self.widget.controls.k_from.setValue(2)
        self.widget.controls.k_to.setValue(6)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_from_to_100_100(self):
        self.widget_from_to()
        self.send_signal(self.widget.Inputs.data, self.d_100_100)
        self.commit_and_wait(wait=100*1000)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_from_to_100_100_no_normalize(self):
        self.widget_from_to()
        self.widget.normalize = False
        self.send_signal(self.widget.Inputs.data, self.d_100_100)
        self.commit_and_wait(wait=100*1000)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_from_to_sampled_silhouette(self):
        self.widget_from_to()
        self.send_signal(self.widget.Inputs.data, self.d_sampled_silhouette)
        self.commit_and_wait(wait=100*1000)

    @benchmark(number=3, warmup=1, repeat=3)
    def bench_wide(self):
        self.widget = self.create_widget(
            OWKMeans, stored_settings={"auto_commit": False})
        self.send_signal(self.widget.Inputs.data, self.d_10_500)
        self.commit_and_wait(wait=100*1000)
