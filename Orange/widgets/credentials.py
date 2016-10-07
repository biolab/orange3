import keyring

SERVICE_NAME = 'Orange3 - {}'


class CredentialManager:
    """
    Class for storage of passwords in the system keyring service.
    All attributes of this class are safely stored.

    Args:
        service_name (str): service name used for storing in keyring.

    Examples:
        >>> cm = CredentialManager('Widget Name')
        >>> cm.username = 'Orange'      # store username
        >>> cm.password = 'Secret'      # store password

        >>> cm.username                 # get username
        'Orange'
        >>> cm.password                 # get password
        'Secret'
    """
    def __init__(self, service_name):
        self.__dict__['__service_name'] = SERVICE_NAME.format(service_name)

    @property
    def service_name(self):
        return self.__dict__['__service_name']

    def __setattr__(self, key, value):
        keyring.set_password(self.service_name, key, value)

    def __getattr__(self, item):
        return keyring.get_password(self.service_name, item)

    def __delattr__(self, item):
        keyring.delete_password(self.service_name, item)
