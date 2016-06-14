import pickle
from unittest import TestCase

from Orange.data import Variable, Domain


class PickleTest(TestCase):
    def setUp(self):
        """ Override __eq__ for Orange objects that do not implement it"""
        self.add_comparator(Domain,
                            compare_members=("attributes", "class_vars",
                                             "class_var", "variables",
                                             "metas", "anonymous"))
        Variable._clear_all_caches()

    old_comparators = {}

    def add_comparator(self, class_, compare_members):
        def compare(self, y):
            for m in compare_members:
                if getattr(self, m) != getattr(y, m):
                    return False
            return True

        def hash(self):
            return "".join(
                [str(getattr(self, m)) for m in compare_members]).__hash__()

        self.old_comparators[class_] = (class_.__eq__, class_.__hash__)
        class_.__eq__ = compare
        class_.__hash__ = hash

    def tearDown(self):
        for c, (eq, hash) in self.old_comparators.items():
            c.__eq__, c.__hash__ = eq, hash

    def assertPicklingPreserves(self, obj):
        for protocol in range(1, pickle.HIGHEST_PROTOCOL + 1):
            obj2 = pickle.loads(pickle.dumps(obj, protocol))
            self.assertEqual(obj, obj2)


def create_pickling_tests(classname, *objs):
    def create_test(descr):
        name, construct_object = descr
        name = "test_{}".format(name)

        def f(self):
            obj = construct_object()
            self.assertPicklingPreserves(obj)

        f.__name__ = name
        return name, f

    tests = dict(map(create_test, objs))
    return type(classname, (PickleTest,), tests)

create_pickling_tests.__test__ = False  # Tell nose this is not a test.
