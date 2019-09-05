import unittest
from Orange.widgets.settings import Context, migrate_str_to_variable


class MigrationsTestCase(unittest.TestCase):
    def test_migrate_str_to_variable(self):
        values = dict(foo=("foo", 1), baz=("baz", 2), qux=("qux", 102), bar=13)

        context = Context(values=values.copy())
        migrate_str_to_variable(context)
        self.assertDictEqual(
            context.values,
            dict(foo=("foo", 101), baz=("baz", 102), qux=("qux", 102), bar=13))

        context = Context(values=values.copy())
        migrate_str_to_variable(context, ("foo", "qux"))
        self.assertDictEqual(
            context.values,
            dict(foo=("foo", 101), baz=("baz", 2), qux=("qux", 102), bar=13))

        context = Context(values=values.copy())
        migrate_str_to_variable(context, "foo")
        self.assertDictEqual(
            context.values,
            dict(foo=("foo", 101), baz=("baz", 2), qux=("qux", 102), bar=13))

        self.assertRaises(KeyError, migrate_str_to_variable, context, "quuux")
