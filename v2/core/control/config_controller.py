class ConfigController:
    def __init__(self, initial_cfg=None):
        self._cfg = initial_cfg or {}
        
    def get(self):
        return self._cfg
    
    def get_value(self, section: str, key: str, default=None):
        if section not in self._cfg:
            if default is not None:
                return default
            raise ValueError(f"missing config section: {section}")
        if key not in self._cfg[section]:
            if default is not None:
                return default
            raise ValueError(f"missing config key: {section}.{key}")
        
        return self._cfg[section][key]

    def set_value(self, section: str, key: str, value):
        if section not in self._cfg:
            self._cfg[section] = {}
        self._cfg[section][key] = value
    
    def update(self, new_cfg: dict):
        for key, value in new_cfg.items():
            if isinstance(value, dict) and key in self._cfg:
                self._cfg[key].update(value)
            else:
                self._cfg[key] = value
        
    def raw(self):
        return self._cfg