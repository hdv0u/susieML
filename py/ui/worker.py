from PyQt5.QtCore import QThread, pyqtSignal

class MLWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self._stop = False
        
    def stop(self):
        self._stop = True
    
    def run(self):
        from main import run_mode
        
        def gui_log(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            self.log.emit(text)
            
        def gui_progress(value):
            self.progress.emit(value)
            
        run_mode(self.mode, log_fn=gui_log, progress_fn=gui_progress, stop_fn=lambda: self._stop)
        self.finished.emit()
    