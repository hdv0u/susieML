from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QRadioButton, QPushButton,
    QVBoxLayout, QButtonGroup, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt
from ui.worker import MLWorker
from core.registry import MODELS

class TestUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test UI")

        self.log = QTextEdit(readOnly=True)
        self.progress = QProgressBar()
        self.status = QLabel("idle")
        
        self.run_btn = QPushButton("Train/Detect")
        self.stop_btn = QPushButton("Stop")
        
        self.run_btn.clicked.connect(self.run)
        self.stop_btn.clicked.connect(self.stop)
        
        layout = QVBoxLayout(self)
        self.mode_group = QButtonGroup(self)
        for mode_id, model_info in MODELS.items():
            rb = QRadioButton(model_info["label"])
            self.mode_group.addButton(rb, int(mode_id))
            layout.addWidget(rb)
            
        layout.addWidget(self.log)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)
        
        self.worker = None
        
    def run(self):
        if self.worker and self.worker.isRunning():
            return
        
        selected_id = self.mode_group.checkedId()
        if selected_id == -1:
            self.log.append("No mode selected")
            return
        
        selected_mode = str(selected_id)
        
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("running")
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        
        self.worker =MLWorker(selected_mode)
        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_done)
        self.worker.start()
        
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("stopping..")
            
    def on_progress(self, value):
        self.progress.setValue(value)
        
    def on_done(self):
        self.status.setText("done")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
            