import unittest

from Orange.widgets.credentials import CredentialManager


class CredentialManagerTests(unittest.TestCase):
    def test_credential_manager(self):
        cm = CredentialManager('Orange')
        cm.key = 'Foo'
        self.assertEqual(cm.key, 'Foo')
        del cm.key
        self.assertEqual(cm.key, None)
