from PyQt5.QtCore import QThread, pyqtSignal

# worker thread for train to avoid blocking the UI 
class TrainWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, trainer, payload, stop_ctrl):
        super().__init__()
        self.payload = payload
        self.stop_ctrl = stop_ctrl
        
    def stop(self):
        self.stop_ctrl.stop()
        
    def run(self):
        self.stop_ctrl.reset()
        # logs and emit them to UI
        def logger(msg):
            self.log.emit(str(msg))
        
        runner = self.payload["runner"]
        runner(self.progress.emit)