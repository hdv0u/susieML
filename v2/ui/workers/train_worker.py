from PyQt5.QtCore import QThread, pyqtSignal

# worker thread for train to avoid blocking the UI 
class TrainWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, trainer, payload):
        super().__init__()
        self.trainer = trainer
        self.payload = payload
        self._stop_flag = False
        
    def stop(self):
        self._stop_flag = True
        
    def run(self):
        # logs and emit them to UI
        def logger(msg):
            self.log.emit(str(msg))
            
        self.trainer.log = logger
        
        dataset_path = self.payload["dataset_path"]
        save_path = self.payload["save_path"]
        
        epochs = self.payload.get("epochs", None)
        
        # send progress updates to the UI using the progress signal
        self.trainer.train(
            dataset_path=dataset_path,
            save_path=save_path,
            stop_flag=lambda: self._stop_flag,
            progress_fn=self.progress.emit,
            override_epochs=epochs
        )