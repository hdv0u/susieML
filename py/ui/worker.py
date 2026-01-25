from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from main import DEBUG
class MLWorker(QThread):
    frame = pyqtSignal(np.ndarray)
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(
        self, mode, model_save=None, train_paths=None, 
        train_labels=None, load_model=None, multi_class=False,
        label_widget=None, parent=None):
        super().__init__()
        self.mode = mode
        self.model_save = model_save
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.load_model = load_model
        self.multi_class = multi_class
        self._stop = False
        
    def stop(self):
        self._stop = True
    
    def run(self):
        from main import run_mode
        def gui_log(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            if DEBUG:
                self.log.emit("debug: " + text)
            else:
                self.log.emit(text)
            
        def gui_progress(value):
            self.progress.emit(value)
            
        run_mode(
            self.mode,
            log_fn=gui_log,
            progress_fn=gui_progress,
            stop_fn=lambda: self._stop,
            frame_fn=lambda f: self.frame.emit(f),
            parent=self,
            model_save=self.model_save,
            train_paths=self.train_paths,
            train_labels=self.train_labels,
            load_model=self.load_model,
            multi_class=self.multi_class,
        )
        self.finished.emit()
    