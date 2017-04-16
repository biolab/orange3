import unittest
import time
import Orange

ORANGE3 = Orange.__version__ >= "3"

class TimeTest(unittest.TestCase):

    def _setUp(self):
         of = object.__getattribute__(self, "setUp")
         r = None
         if of:
             r = of()
         self._startTime = time.time()
         return r

    def _tearDown(self):
         t = time.time() - self._startTime
         print("")
         print("TIMING %s: %.6f" % (self.id(), t))
         of = object.__getattribute__(self, "tearDown")
         r = None
         if of:
             return of()
  
    def __getattribute__(self, attr):
        if attr == "setUp":
            return self._setUp
        if attr == "tearDown":
            return self._tearDown
        else:   
            return object.__getattribute__(self, attr)

