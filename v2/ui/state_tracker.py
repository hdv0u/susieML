class StateTracker:
    def __init__(self):
        self._last_state = {}
        self._enabled = True
        
    def capture(self, ui):
        return {
            "epochs": ui.gen.value(),
            "lr": ui.lr.value(),
            "arch_depth": ui.arch_depth.value(),

            "side_len": ui.side_len.value(),
            "steps": ui.steps.value(),
            "threshold": ui.threshold.value(),

            "augment_count": ui.augment_count.value(),
            "flip": ui.horizontal_flip.isChecked(),
            "brightness": ui.brightness_aug.isChecked(),
            "shift": ui.rotation_aug.isChecked(),
        }
    
    # compares current state to last, logs and updates last state
    def diff(self, ui):
        if not self._enabled:
            return
        
        current = self.capture(ui)
        
        if not self._last_state:
            self._last_state = current.copy()
            print("[Tracker] initialized base state")
            return
        
        for key, new_val in current.items():
            old_val = self._last_state.get(key)
            
            if new_val != old_val:
                print(f"[TrackerChange] {key}: {old_val} -> {new_val}")

        self._last_state = current.copy()
        
    def reset(self):
        self._last_state = {}
        
    def enable(self, state: bool):
        self._enabled = True
        