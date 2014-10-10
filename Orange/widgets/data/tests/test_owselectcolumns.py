from unittest import TestCase
from Orange.data import Table
from Orange.widgets.data.owselectcolumns import \
    SelectAttributesDomainContextHandler
from Orange.widgets.tests.test_settings import MockWidget


class TestSelectAttributesDomainContextHandler(TestCase):
    def setUp(self):
        self.handler = SelectAttributesDomainContextHandler()
        self.handler.read_defaults = lambda: None  # Disable reading settings from disk
        self.handler.bind(MockWidget)

    def test_clone_context(self):
        context = self.handler.new_context()
        iris = Table('iris')
        attrs, metas = self.handler.encode_domain(iris.domain)
        self.handler.clone_context(context, iris.domain, attrs, metas)
