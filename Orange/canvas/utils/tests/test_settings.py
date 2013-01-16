"""
Tests for settings utility module.

"""
import logging

from PyQt4.QtCore import QSettings


from ...gui import test


class TestUserSettings(test.QAppTestCase):
    def setUp(self):
        logging.basicConfig()
        test.QAppTestCase.setUp(self)

    def test_settings(self):
        from ..settings import Settings, config_slot

        spec = [config_slot("foo", bool, True, "foo doc"),
                config_slot("bar", int, 0, "bar doc"),
                ]

        store = QSettings(QSettings.IniFormat, QSettings.UserScope,
                          "biolab.si", "Orange Canvas Unit Tests")
        store.clear()
        settings = Settings(defaults=spec, store=store)

        self.assertEqual(settings["foo"], True)
        self.assertEqual(settings.get("foo"), True)

        self.assertEqual(settings.get("bar", 3), 0, "Defaults")
        self.assertEqual(settings.get("bar"), settings["bar"])

        self.assertEqual(settings.get("does not exist", "^&"), "^&",
                         "get with default")

        self.assertIs(settings.get("does not exist"), None)

        with self.assertRaises(KeyError):
            settings["does not exist"]

        self.assertTrue(settings.isdefault("foo"))

        changed = []

        settings.valueChanged.connect(
            lambda key, value: changed.append((key, value))
        )

        settings["foo"] = False
        self.assertEqual(changed[-1], ("foo", False), "valueChanged signal")
        self.assertEqual(len(changed), 1)
        self.assertEqual(settings["foo"], False, "updated value")
        self.assertEqual(settings.get("foo"), False)
        self.assertFalse(settings.isdefault("foo"))

        settings["bar"] = 1
        self.assertEqual(changed[-1], ("bar", 1), "valueChanged signal")
        self.assertEqual(len(changed), 2)
        self.assertEqual(settings["bar"], 1)
        self.assertFalse(settings.isdefault("bar"))

        del settings["bar"]
        self.assertEqual(settings["bar"], 0)
        self.assertEqual(changed[-1], ("bar", 0))

        # Only str or unicode can be keys
        with self.assertRaises(TypeError):
            settings[1] = 3

        # value type check
        with self.assertRaises(TypeError):
            settings["foo"] = []

        self.assertEqual(len(changed), 3)

        settings.add_default_slot(config_slot("foobar/foo", object, None, ""))

        group = settings.group("foobar")

        self.assertIs(group["foo"], None)
        group["foo"] = 3

        self.assertEqual(changed[-1], ("foobar/foo", 3))

        group["foonew"] = 5
        self.assertIn("foobar/foonew", settings)

        settings["newkey"] = "newkeyvalue"
        self.assertIn("newkey", settings)

        group1 = group.group("bar")
        group1["barval"] = "barval"

        self.assertIn("foobar/bar/barval", settings)

        settings["foobar/bar/barval"] = 5
        self.assertEqual(changed[-1], ("foobar/bar/barval", 5))

        settings.clear()
        self.assertSetEqual(set(settings.keys()),
                            set(["foo", "bar", "foobar/foo"]))
