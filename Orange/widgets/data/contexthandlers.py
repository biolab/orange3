from Orange.widgets.settings import DomainContextHandler


class SelectAttributesDomainContextHandler(DomainContextHandler):
    """Select Columns widget has context settings in a specific format.
    This context handler modifies match and clone_context to account for that.
    """

    def match_value(self, setting, value, attrs, metas):
        if setting.name == 'domain_role_hints':
            value = self.decode_setting(setting, value)
            matched = available = 0
            for item, category in value.items():
                role, role_idx = category
                if role != 'available':
                    available += 1
                    if self._var_exists(setting, item, attrs, metas):
                        matched += 1
            return matched, available
        return super().match_value(setting, value, attrs, metas)

    def filter_value(self, setting, data, domain, attrs, metas):
        value = data.get(setting.name, None)
        value = self.decode_setting(setting, value)

        if isinstance(value, dict):
            for item, category in list(value.items()):
                if not self._var_exists(setting, item, attrs, metas):
                    del value[item]
        else:
            super().filter_value(setting, data, domain, attrs, metas)
