import pickle
import unittest

class PickleTest(unittest.TestCase):
    DEBUG = False

    def test_pickling(self):
        if not hasattr(self, "generate_objects"):
            unittest.skip("Method generate_objects does not exist.")

        for obj in self.generate_objects():
            if self.DEBUG:
                attrs = obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__
                for attr in attrs:
                    print(attr)
                    pickle.dumps(attr)

            obj2 = pickle.loads(pickle.dumps(obj))
            self.assertEqual(obj, obj2)
