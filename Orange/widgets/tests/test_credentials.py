import unittest
from unittest.mock import patch

from Orange.widgets.credentials import CredentialManager


class TestCredentialManager(unittest.TestCase):
    def setUp(self):
        self.cm = CredentialManager('Orange')
        self.cm.key = "Foo"

    def test_credential_manager(self):
        cm = CredentialManager('Orange')
        cm.key = 'Foo'
        self.assertEqual(cm.key, 'Foo')
        del cm.key
        self.assertEqual(cm.key, None)

    def test_set_password(self):
        """
        Handle error when setting password fails.
        GH-2354
        """
        with patch("keyring.set_password", side_effect=Exception):
            self.cm.key = ""

    def test_delete_password(self):
        """
        Handling error when deleting password fails
        GH-2354
        """
        with patch("keyring.delete_password", side_effect=Exception):
            del self.cm.key

    def test_get_password(self):
        """
        Handling errors when getting password fails.
        GH-2354
        """
        with patch("keyring.get_password", side_effect=Exception):
            self.assertEqual(self.cm.key, None)
