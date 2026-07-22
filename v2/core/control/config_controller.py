class ConfigController:
    def __init__(self, initial_cfg=None):
        self._cfg = initial_cfg or {}
        
    def get(self):
        return self._cfg
    
    def get_value(self, section: str, key: str, default=None):
        if section not in self._cfg:
            raise ValueError(f"Section '{section}' not found in config")
        if key not in self._cfg[section]:
            if default is not None:
                return default
            raise ValueError(f"Key '{key}' not found in section '{section}'")
        return self._cfg[section][key]

    def set_value(self, section: str, key: str, value):
        if section not in self._cfg:
            self._cfg[section] = {}
        self._cfg[section][key] = value
        
    def raw(self):
        return self._cfg