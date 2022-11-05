import unittest
from Orange.widgets.utils.localization import *


class TestEn(unittest.TestCase):
    def test_pl(self):
        self.assertEqual(pl(0, "cat"), "cats")
        self.assertEqual(pl(1, "cat"), "cat")
        self.assertEqual(pl(2, "cat"), "cats")
        self.assertEqual(pl(100, "cat"), "cats")
        self.assertEqual(pl(101, "cat"), "cats")

        self.assertEqual(pl(0, "cat|cats"), "cats")
        self.assertEqual(pl(1, "cat|cats"), "cat")
        self.assertEqual(pl(2, "cat|cats"), "cats")
        self.assertEqual(pl(100, "cat|cats"), "cats")
        self.assertEqual(pl(101, "cat|cats"), "cats")


class TestSi(unittest.TestCase):
    def test_plsi_4(self):
        self.assertEqual(plsi(0, "okno|okni|okna|oken"), "oken")
        self.assertEqual(plsi(1, "okno|okni|okna|oken"), "okno")
        self.assertEqual(plsi(2, "okno|okni|okna|oken"), "okni")
        self.assertEqual(plsi(3, "okno|okni|okna|oken"), "okna")
        self.assertEqual(plsi(4, "okno|okni|okna|oken"), "okna")
        self.assertEqual(plsi(5, "okno|okni|okna|oken"), "oken")
        self.assertEqual(plsi(11, "okno|okni|okna|oken"), "oken")
        self.assertEqual(plsi(100, "okno|okni|okna|oken"), "oken")
        self.assertEqual(plsi(101, "okno|okni|okna|oken"), "okno")
        self.assertEqual(plsi(102, "okno|okni|okna|oken"), "okni")
        self.assertEqual(plsi(103, "okno|okni|okna|oken"), "okna")
        self.assertEqual(plsi(105, "okno|okni|okna|oken"), "oken")
        self.assertEqual(plsi(1001, "okno|okni|okna|oken"), "okno")

    def test_plsi_3(self):
        self.assertEqual(plsi(0, "oknu|oknoma|oknom"), "oknom")
        self.assertEqual(plsi(1, "oknu|oknoma|oknom"), "oknu")
        self.assertEqual(plsi(2, "oknu|oknoma|oknom"), "oknoma")
        self.assertEqual(plsi(3, "oknu|oknoma|oknom"), "oknom")
        self.assertEqual(plsi(5, "oknu|oknoma|oknom"), "oknom")
        self.assertEqual(plsi(1, "oknu|oknoma|oknom"), "oknu")
        self.assertEqual(plsi(105, "oknu|oknoma|oknom"), "oknom")

    def test_plsi_1(self):
        self.assertEqual(plsi(0, "miza"), "miz")
        self.assertEqual(plsi(1, "miza"), "miza")
        self.assertEqual(plsi(2, "miza"), "mizi")
        self.assertEqual(plsi(3, "miza"), "mize")
        self.assertEqual(plsi(5, "miza"), "miz")
        self.assertEqual(plsi(101, "miza"), "miza")
        self.assertEqual(plsi(105, "miza"), "miz")

        self.assertEqual(plsi(0, "primer"), "primerov")
        self.assertEqual(plsi(1, "primer"), "primer")
        self.assertEqual(plsi(2, "primer"), "primera")
        self.assertEqual(plsi(3, "primer"), "primeri")
        self.assertEqual(plsi(5, "primer"), "primerov")
        self.assertEqual(plsi(50, "primer"), "primerov")
        self.assertEqual(plsi(101, "primer"), "primer")
        self.assertEqual(plsi(105, "primer"), "primerov")

    def test_plsi_sz(self):
        for propn in "z0 z1 z2 s3 s4 s5 s6 s7 z8 z9 z10 " \
                      "z11 z12 s13 s14 s15 s16 s17 z18 z19 z20 " \
                      "z21 z22 s23 z31 z32 s35 s40 s50 s60 s70 z80 z90 " \
                      "z200 z22334 s3943 z832492 " \
                      "s100 s108 s1000 s13333 s122222 z1000000 " \
                      "z1000000000 z1000000000000".split():
            self.assertEqual(plsi_sz(int(propn[1:])), propn[0], propn)


if __name__ == "__main__":
    unittest.main()