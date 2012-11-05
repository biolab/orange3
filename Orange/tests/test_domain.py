import unittest

from Orange import testing

from Orange import data

class PickleDomain(testing.PickleTest):
    def generate_objects(self):
        yield data.Domain([])

    def assertEqual(self, first, second, msg=None):
        super().assertEqual(first.attributes, second.attributes)
        super().assertEqual(first.anonymous, second.anonymous)





class TestDomain(unittest.TestCase):
    attributes = ["Feature %i" % i for i in range(10)]
    class_vars = ["Class %i" % i for i in range(3)]
    metas = ["Meta %i" % i for i in range(5)]
    attr_vars = [data.ContinuousVariable(name=a) if isinstance(a, str) else a for a in attributes]
    class_vars = [data.ContinuousVariable(name=c) if isinstance(c, str) else c for c in class_vars]
    meta_vars = [data.StringVariable(name=m) if isinstance(m, str) else m for m in metas]


    def test_can_pickle_domain(self):
        domain = data.Domain(self.attr_vars)
        pickle.dumps(domain)