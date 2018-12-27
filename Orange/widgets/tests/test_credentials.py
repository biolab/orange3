import unittest
from unittest.mock import patch

import keyring
import keyring.errors
import keyring.backend

from Orange.widgets.credentials import CredentialManager


# minimal in-memory keyring implementation so the test is not dependent on
# the system config/services.
class Keyring(keyring.backend.KeyringBackend):
    priority = 0

    def __init__(self):
        self.__store = {}

    def set_password(self, service, username, password=None):
        self.__store[(service, username)] = password

    def get_password(self, service, username):
        return self.__store.get((service, username), None)

    def delete_password(self, service, username):
        try:
            del self.__store[service, username]
        except KeyError:
            raise keyring.errors.PasswordDeleteError()


class TestCredentialManager(unittest.TestCase):
    def setUp(self):
        self._ring = keyring.get_keyring()
        keyring.set_keyring(Keyring())
        self.cm = CredentialManager('Orange')

    def tearDown(self):
        keyring.set_keyring(self._ring)

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
