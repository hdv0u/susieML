class RunController:
    def __init__(self):
        self._running = True
        self._paused = False
        
    def stop(self):
        self._running = False
        
    def pause(self):
        self._paused = True
        
    def resume(self):
        self._paused = False
    
    def reset(self):
        self._running = True
        self._paused = False
        
    def is_running(self):
        return self._running
    
    def is_paused(self):
        return self._paused