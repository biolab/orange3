from Orange import testing
from Orange import data

class PickleDomain(testing.PickleTest):
    def generate_objects(self):
        yield data.Domain([])

    def assertEqual(self, first, second, msg=None):
        super().assertEqual(first.attributes, second.attributes)
        super().assertEqual(first.anonymous, second.anonymous)
