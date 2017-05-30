import logging

import keyring

SERVICE_NAME = 'Orange3 - {}'

log = logging.getLogger(__name__)


class CredentialManager:
    """
    Class for storage of passwords in the system keyring service.
    All attributes of this class are safely stored.

    Args:
        service_name (str): service name used for storing in keyring.

    Examples:
        >>> cm = CredentialManager('Widget Name')
        >>> cm.some_secret = 'api-key-1234'
        >>> cm.some_secret
        'api-key-1234'
        >>> del cm.some_secret
        >>> cm.some_secret
    """
    def __init__(self, service_name):
        self.__dict__['__service_name'] = SERVICE_NAME.format(service_name)

    @property
    def service_name(self):
        return self.__dict__['__service_name']

    def __setattr__(self, key, value):
        try:
            keyring.set_password(self.service_name, key, value)
        except Exception:
            log.exception("Failed to set secret '%s' of '%r'.",
                          key, self.service_name)

    def __getattr__(self, item):
        try:
            return keyring.get_password(self.service_name, item)
        except Exception:
            log.exception("Failed to get secret '%s' of '%r'.",
                          item, self.service_name)

    def __delattr__(self, item):
        try:
            keyring.delete_password(self.service_name, item)
        except Exception:
            log.exception("Failed to delete secret '%s' of '%r'.",
                          item, self.service_name)
